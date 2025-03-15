import torch
import torchaudio
from torchaudio.transforms import MFCC
import glob
import os
from tqdm import tqdm


def preprocess_audio(audio_path, audio_length=256):
    waveform, sr = torchaudio.load(audio_path)

    waveform = waveform - waveform.mean()
    norm_mean = -4.503877
    norm_std = 5.141276

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    
    n_frames = fbank.shape[0]
    
    p = audio_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:audio_length, :]
    fbank = (fbank - norm_mean) / (norm_std * 2)

    fbank = fbank.unsqueeze(0)

    return fbank, n_frames

if __name__ == '__main__':
    audio_paths = glob.glob('AudioWAV/*.wav')
    # print(audio_paths)
    os.makedirs('fbank', exist_ok=True)
    n_flist = []
    for audio_path in tqdm(audio_paths):
        fbank, n_frames = preprocess_audio(audio_path)
        audio_name = os.path.basename(audio_path)[:-4]
        torch.save(fbank, f'fbank/{audio_name}.pt')
        n_flist.append(n_frames)
    
    with open('n_frames.txt', 'w') as f:
        f.write(str(n_flist))
    