import torch
import stft
import conversions


class Reconstructor(torch.nn.Module):
    def __init__(self, net, log_frac=None):
        super(Reconstructor, self).__init__()
        self.net = net
        self.forward = self.reconst
        self.log_frac = log_frac

    def reconst(self, latent, metadata):
        return self.net(latent, metadata)

    def reconst_loss(self, reconst, original, reconst_exp=None):
        if reconst_exp is None:
            reconst_exp = torch.exp(reconst)
        original_exp = torch.exp(original)

        total_kl = (
            original_exp * (original - reconst) -
            original_exp + reconst_exp).sum()
        total_is = (
            (original_exp / reconst_exp) - (original - reconst) - 1).sum()
        total_other = (total_kl + total_is) / 2.0
        total_log = ((reconst - original) ** 2).sum()

        return (total_other * (1.0 - self.log_frac) +
                total_log * self.log_frac) / reconst.numel()


class PhaseMatcher(torch.nn.Module):
    def __init__(self, sample_rate, conv_options, iters=1):
        super(PhaseMatcher, self).__init__()
        segment_length = int(round(sample_rate * conv_options.window_length))
        filter_length = conv_options.fft_size
        hop_length = int(round(sample_rate * conv_options.interval_length))

        self.conv_options = conv_options
        self.stft = stft.STFT(
            segment_length=segment_length,
            filter_length=filter_length,
            hop_length=hop_length)
        self.iters = iters

    def match_to_phase(self, reconst, phase):
        width = reconst.shape[2]

        reconst_exp = conversions.from_log(
            reconst, options=self.conv_options,
            make_pos_fn=lambda val: val.clamp(min=0.0),
            exp_fn=torch.exp)

        for _ in range(self.iters):
            samples = self.stft.inverse(reconst_exp, phase)
            reconst_exp, _ = self.stft.transform(samples)
            reconst_exp = reconst_exp[:, :, :width]

        reconst = conversions.to_log(
            reconst_exp.clamp(min=0.0), options=self.conv_options,
            log_fn=torch.log)

        return reconst
