
import torch
import numpy as np
import PIL
from PIL import Image
from scipy import integrate


from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.autoencoder import AutoencoderKL


#VAE模型
def load_vae(ckpt):
    #初始化模型
    init_config = {
        "ddconfig":{
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult":[1,2,4,4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0,
        },
        "embed_dim": 4
    }
    vae = AutoencoderKL(**init_config)
    #加载预训练参数
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model_dict = vae.state_dict()
    for k, v in model_dict.items():
        model_dict[k] = sd["first_stage_model."+k]
    vae.load_state_dict(model_dict, strict=False)

    vae.eval()
    return vae

def test_vae(ckpt):
    vae = load_vae(ckpt)
    img = load_image("assets/test.png")     #(1,3,512,512)   
    latent = vae.encode(img).sample()       #(1,4,64,64)
    samples = vae.decode(latent)            #(1,3,512,512)
    save_image(samples,"output/vae.png")

#载入图片
def load_image(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0   #(512, 512, 3)
    image = image[None].transpose(0, 3, 1, 2)           # (1, 3, 512, 512)
    image = torch.from_numpy(image)
    return 2.*image - 1.

#保存图片
def save_image(samples, path):     
    samples = 255 * (samples/2+0.5).clamp(0,1)    # (1, 3, 512, 512)
    samples = samples.detach().numpy()
    samples = samples.transpose(0, 2, 3, 1)       #(1, 512, 512, 3)
    image = samples[0]                            #(512, 512, 3)
    image = Image.fromarray(image.astype(np.uint8))
    image.save(path)


def load_unet(ckpt):
    unet_init_config = {
            "in_channels": 4,
            "out_channels": 4,
            "model_channels": 320,
            "attention_resolutions": [ 4, 2, 1 ],
            "num_res_blocks": 2,
            "channel_mult": [ 1, 2, 4, 4 ],
            "num_head_channels": 64,
            "use_spatial_transformer": True,
            "transformer_depth": 1,
            "context_dim": 1024,
            "use_checkpoint": True,
    }
    unet = UNetModel(**unet_init_config)
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]

    model_dict = unet.state_dict()
    for k, v in model_dict.items():
        model_dict[k] = sd["model.diffusion_model."+k]

    unet.load_state_dict(model_dict, strict=False)
    unet.cuda()
    unet.eval()

    return unet


def test_unet(ckpt):
    #vae
    latent = torch.randn(1,4,64,64).cuda()
    #text
    text_embeddings =torch.randn(1, 77, 768).cuda()
    #timestamp
    timestamp = torch.tensor([0]).cuda()
    unet = load_unet(ckpt)
    y = unet(latent.cuda(), timestamp.cuda(), text_embeddings.cuda())
    print(y.shape) #(1, 4, 64, 64)


class lms_scheduler():
    def __init__(self):
        beta_start = 0.00085
        beta_end = 0.012
        num_train_timesteps = 1000

        #betas = [9.99999975e-05 1.19919918e-04 1.39839845e-04 1.59759758e-04 ...
        self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2   
        #alphas = #[0.9999     0.9998801  0.99986017 ...
        self.alphas = 1.0 - self.betas   
        # alphas_cumprod=累积乘积 [9.99899983e-01 9.99780059e-01 9.99640286e-01 9.99480605e-01 ...
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0) 
        return

    def set_timesteps(self, num_inference_steps=100):
        self.num_inference_steps = num_inference_steps
        #1000：num_train_timesteps
        self.timesteps = np.linspace(1000 - 1, 0, num_inference_steps, dtype=float)  #[999.         988.90909091 978.81818182 968.72727273 958.63636364 …… ] 100个
        low_idx = np.floor(self.timesteps).astype(int) #[999 988 978 968 958  ...] 100个
        high_idx = np.ceil(self.timesteps).astype(int) #[999 989 979 969 959  ...]  100个
        frac = np.mod(self.timesteps, 1.0)             #[0.         0.90909091 0.81818182 0.72727273 ... ] 小数部分

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)  #[1.00013297e-02 1.48320440e-02  1000个
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]  #[1.57407227e+02 1.42219348e+02   100个
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32) #最后加个零 101个
        self.derivatives = []

    def get_lms_coefficient(self, order, t, current_order):
        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
            return prod

        integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]

        return integrated_coeff

    def step(self,model_output,timestep,sample):
        order = 4
        sigma = self.sigmas[timestep]
        pred_original_sample = sample - sigma * model_output
        derivative = (sample - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)
        order = min(timestep + 1, order)
        lms_coeffs = [self.get_lms_coefficient(order, timestep, curr_order) for curr_order in range(order)]    
        prev_sample = sample + sum(coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives)))
        return prev_sample


# def txt2img():
#     #unet
#     unet = UNetModel("models/Stable-diffusion/sd-v1-2.ckpt")

#     #调度器
#     scheduler = lms_scheduler()
#     scheduler.set_timesteps(100)

#     #文本编码
#     prompts = ["a photograph of an astronaut riding a horse"]
#     text_embeddings = prompts_embedding(prompts)
#     text_embeddings = text_embeddings.cuda()     #(1, 77, 768)
#     uncond_prompts = [""]
#     uncond_embeddings = prompts_embedding(uncond_prompts)
#     uncond_embeddings = uncond_embeddings.cuda() #(1, 77, 768)

#     #初始隐变量
#     latents = torch.randn( (1, 4, 64, 64))  #(1, 4, 64, 64)
#     latents = latents * scheduler.sigmas[0]    #sigmas[0]=157.40723
#     latents = latents.cuda()

#     #循环步骤
#     for i, t in enumerate(scheduler.timesteps):  #timesteps=[999.  988.90909091 978.81818182 ...100个
#         latent_model_input = latents  #(1, 4, 64, 64)  
#         sigma = scheduler.sigmas[i]
#         latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
#         timestamp = torch.tensor([t]).cuda()

#         with torch.no_grad():  
#             noise_pred_text = unet(latent_model_input, timestamp, text_embeddings)
#             noise_pred_uncond = unet(latent_model_input, timestamp, uncond_embeddings)
#             guidance_scale = 7.5 
#             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#             latents = scheduler.step(noise_pred, i, latents)
        
#     vae = load_vae()
#     latents = 1 / 0.18215 * latents
#     image = vae.decode(latents.cpu())  #(1, 3, 512, 512)
#     save_image(image,"txt2img.png")


if __name__ == "__main__":
    
    # test_vae("models/Stable-diffusion/sd-v1-2.ckpt")

    test_unet("models/Stable-diffusion/sd-v1-2.ckpt")