import torch
from torch import nn
import modules.Transforms as ts
from functools import partial
from inspect import getfullargspec
from compressai.models import MeanScaleHyperprior
from compressai.layers import  MaskedConv2d
from compressai.entropy_models import GaussianConditional
import torchvision.transforms as T
from torchvision.utils import save_image 

class GoogleHyperPrior(MeanScaleHyperprior):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, 
                 num_filters=128, num_features=128, num_hyperpriors=128, 
                 downsample_8=False, **kwargs):
        super().__init__(num_filters, num_features, **kwargs)

        self.g_a = ts.GoogleAnalysisTransform(in_channels, num_features, num_filters, kernel_size, downsample_8)
        self.g_s = ts.GoogleSynthesisTransform(out_channels, num_features, num_filters, kernel_size, downsample_8)
        self.h_a = ts.GoogleHyperAnalysisTransform(num_features, num_filters, num_hyperpriors)
        if 'mask_rate' in kwargs:
            self.h_s = nn.Sequential(
                ts.MaskLayer(kwargs['mask_rate']),
                ts.GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)
            )
        else:
            self.h_s = ts.GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)

    def forward(self, x):

        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        # print("hyper y", y.requires_grad)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }



class SwinGSHyperPrior(MeanScaleHyperprior):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, 
                 num_filters=128, num_features=128, num_hyperpriors=128, 
                 downsample_8=False, embed_ws=2, input_resolution=(256, 256),
                 num_heads=8, window_size=8, mlp_ratio=4., depth=1, **kwargs):
        super().__init__(num_filters, num_features, **kwargs)

        self.g_a = ts.GoogleAnalysisTransform(in_channels, num_features, num_filters, kernel_size, downsample_8, True)
        self.g_s = ts.GoogleSynthesisTransform(out_channels, num_features, num_filters, kernel_size, downsample_8, False)
        self.h_a = ts.GoogleHyperAnalysisTransform(num_features, num_filters, num_hyperpriors)
        self.h_s = ts.GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)

        input_resolution = tuple([s//embed_ws for s in input_resolution])
        self.embed_x = ts.TokenEmbedding(in_channels, num_filters, embed_ws)
        # self.embed_xc = ts.TokenEmbedding(in_channels, num_filters, embed_ws)
        self.xc_window_embed = ts.WindowEmbed(num_filters, window_size, 0)
        self.xc_kv = nn.Linear(num_filters, num_filters * 2, bias=True)

        self.embed_ws = embed_ws
        self.num_filters = num_filters
        self.swin_gr = ts.SwinGeneralResidual(depth, num_filters, input_resolution, num_heads, window_size, 
                                                mlp_ratio, cross=True, cancel_out=False)
        # self.swin_gs = ts.SwinGeneralSum(depth, num_filters, input_resolution, num_heads, window_size, 
        #                                     mlp_ratio, cross=True, cancel_out=False)
        # self.unembed = ts.TokenUnEmbedding(num_filters, out_channels, embed_ws)

    def forward(self, x, xc):
        x_size = tuple([s//self.embed_ws for s in x.size()[-2:]])

        x_token = self.embed_x(x)
        xc_token = self.embed_x(xc)
        # xc_token = self.embed_xc(xc)
        # print(x_size, x_token.size(), xc_token.size())
        k, v = self.xc_kv(self.xc_window_embed(xc_token, x_size)).split(self.num_filters, dim=-1)
        # print(k.size(), v.size())
        gr_token, tokenc = self.swin_gr(x_token, k, v, x_size)
        # gr_token = self.swin_gr(x_token - xc_token, None, None, x_size)

        outputs = super().forward(gr_token)
        outputs['x_token'] = x_token
        outputs['xc_token'] = xc_token
        outputs['tokenc'] = tokenc
        outputs['gr_token'] = gr_token
        # x_hat_token = self.swin_gs(outputs['x_hat'], k, v, x_size)
        # x_hat = self.unembed(x_hat_token)
        # return {
        #     "x_hat": x_hat,
        #     "likelihoods": outputs["likelihoods"],
        # }
        return outputs

class GoogleHyperPrior_autoregressive(MeanScaleHyperprior):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, 
                 num_filters=128, num_features=128, num_hyperpriors=128, 
                 downsample_8=False, **kwargs):
        super().__init__(num_filters, num_features, **kwargs)

        self.g_a = ts.GoogleAnalysisTransform(in_channels, num_features, num_filters, kernel_size, downsample_8)
        self.g_s = ts.GoogleSynthesisTransform(out_channels, num_features, num_filters, kernel_size, downsample_8)
        self.h_a = ts.GoogleHyperAnalysisTransform(num_features, num_filters, num_hyperpriors)
        
        # for context model
        self.N = int(num_features)
        self.M = int(num_hyperpriors)
        self.context_prediction = MaskedConv2d(
            self.M, 2 * self.M, kernel_size=5, padding=2, stride=1
        )    
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.M * 12 // 3, self.M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, self.M * 6 // 3, 1),
        )

        # for mask
        if 'mask_rate' in kwargs:
            self.h_s = nn.Sequential(
                ts.MaskLayer(kwargs['mask_rate']),
                ts.GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)
            )
        else:
            self.h_s = ts.GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)

    def forward(self, x):

        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        print("params: ",params.shape)

        # print("ha: ", self.ha.requires_grad)
        # print("context_prediction: ", self.context_prediction.requires_grad)
        # print("auto y: ", y.requires_grad)  

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )         
        # print("yhat: ", y_hat.shape)
        # print("auto y_hat: ", y_hat.requires_grad)  
        ctx_params = self.context_prediction(y_hat)
        print("ctx_params: ", ctx_params.shape)
        # print("auto ctx_params: ",ctx_params.requires_grad)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        print("gaussian_params: ", gaussian_params.shape)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # print("y_likelihoods: ",y_likelihoods.requires_grad)


        x_hat = self.g_s(y_hat)
        # print("x_hat: ", x_hat.requires_grad)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

class Channel_wise_context(MeanScaleHyperprior):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, 
                 num_filters=128, num_features=128, num_hyperpriors=128, 
                 downsample_8=False, number_group=4, **kwargs):
        super().__init__(num_filters, num_features, **kwargs)

        self.num_features = num_features
        self.num_hyperpriors = num_hyperpriors

        assert num_features % number_group == 0, "channel can be cut into 4 parts equally"
        self.num_group = number_group
        self.equal_slice = num_features // number_group

        self.g_a = ts.GoogleAnalysisTransform(in_channels, num_features, num_filters, kernel_size, downsample_8)
        self.g_s = ts.GoogleSynthesisTransform(out_channels, num_features, num_filters, kernel_size, downsample_8)
        self.h_a = ts.GoogleHyperAnalysisTransform(num_features, num_filters, num_hyperpriors)
        if 'mask_rate' in kwargs:
            self.h_s = nn.Sequential(
                ts.MaskLayer(kwargs['mask_rate']),
                ts.GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)
            )
        else:
            self.h_s = ts.GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)

        # self.lpr = nn.ModuleList(
        #     nn.Sequential(
        #         nn.Conv2d(self.num_hyperpriors + i * self.equal_slice + self.equal_slice, self.equal_slice * 4, 3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(self.equal_slice * 4, self.equal_slice * 2, 3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(self.equal_slice * 2, self.equal_slice, 3, padding=1),
        #     ) for i in range(self.num_group)
        # )


        self.mu_predictor = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.num_hyperpriors + i * self.equal_slice, self.equal_slice * 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.equal_slice * 4, self.equal_slice * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.equal_slice * 2, self.equal_slice, 3, padding=1),
            ) for i in range(self.num_group)
        )

        self.sigma_predictor = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.num_hyperpriors + i * self.equal_slice, self.equal_slice * 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.equal_slice * 4, self.equal_slice * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.equal_slice * 2, self.equal_slice, 3, padding=1),
            ) for i in range(self.num_group)
        )

    def forward(self, x):
        # transform = T.ToPILImage()
        # save_image(x[0], f'res_img/origin.png')
        # transform(x[0]).convert('RGB').save(f'res_img/origin.png')

        y = self.g_a(x) # B, 128, 16, 16
        z = self.h_a(y) # B, 128, 4, 4
        # print("z: ",z.requires_grad )
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # print("y: ", y.shape)
        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")  
        y_slices = y.split([self.equal_slice for _ in range(self.num_group)], 1)
        y_hat_slices = y_hat.split([self.equal_slice for _ in range(self.num_group)], 1) 

        mu_hat_list = []
        prob_list = []

        for slice_idx, y_hat_slice in enumerate(y_hat_slices):
            # print("means_hat: ",means_hat.requires_grad)
            new_mu_slice = torch.cat([means_hat] + mu_hat_list, dim=1)
            # print("new_mu_slice: ",new_mu_slice.requires_grad)
            new_sigma_slice = torch.cat([scales_hat] + mu_hat_list, dim=1)

            mu = self.mu_predictor[slice_idx](new_mu_slice)
            # print("mu: ",mu.requires_grad)
            sigma = self.sigma_predictor[slice_idx](new_sigma_slice)
            # print("sigma: ",sigma.requires_grad)

            _, y_likelihoods = self.gaussian_conditional(y_slices[slice_idx], sigma, means=mu)
            prob_list.append(y_likelihoods)
            
            
            # print("y_like: ", y_likelihoods.requires_grad)
            lpr_input = torch.cat((new_mu_slice, y_hat_slice), dim=1)
            # print("lpr_input: ", lpr_input.requires_grad)
            res = self.lpr[slice_idx](lpr_input)
            # img = transform(res.sum(dim=1)[0]).save(f'res_img/seq_{slice_idx}.png')
            # print(res.shape)
            # print("res: ", res.requires_grad)
            new_y_hat = res + y_hat_slice
            # save_image(new_y_hat.sum(dim=1)[0], f'res_img/seq_{slice_idx}_hat.png')
            mu_hat_list.append(new_y_hat)
            mu_hat_list.append(y_likelihoods)


        y_likelihoods = torch.cat(prob_list, dim=1)
        x_hat = self.g_s(y_hat)
        # print("x_hat: ", x_hat.requires_grad)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }




__CODER_TYPES__ = {
    "GoogleHyperPriorCoder": GoogleHyperPrior,
    "SwinGSHyperPrior": SwinGSHyperPrior,
    "GoogleHyperPrior_autoregressive" : GoogleHyperPrior_autoregressive,
    "Channel_wise_context" : Channel_wise_context
}

def get_coder_from_args(args):
    # print("arch: ",args.architecture)
    coder = __CODER_TYPES__[args.architecture]
    # kwargs = {k: v for k, v in vars(args).items() if k in getfullargspec(coder.__init__).args}
    kwargs = vars(args)
    coder = partial(coder, **kwargs)
    return coder