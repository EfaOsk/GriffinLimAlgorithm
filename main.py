import torch
import scipy.signal
from scipy.io import wavfile
import numpy as np



def griffin_lim(magnitude_spectrogram, iterations=30, n_fft=1024, hop_length=None, win_length=None, length=None):
    """
    Implements the Griffin-Lim algorithm to estimate the phase given only the magnitude
    of the Short-Time Fourier Transform (STFT).

    Args:
    magnitude_spectrogram (torch.Tensor): Magnitude spectrogram
    [..]
    Returns:
    torch.Tensor: Reconstructed time-domain signal
    """

    # Randomly initialize
    phase = torch.exp(2j * torch.pi * torch.rand_like(magnitude_spectrogram))
    complex_spectrogram = magnitude_spectrogram * phase

    # Window function for the STFT/ISTFT
    window = torch.hann_window(win_length)

    for _ in range(iterations):
        # Inverse STFT
        signal = torch.istft(complex_spectrogram, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length)

        # Re-calculate STFT with return_complex=True
        recomplex_spectrogram = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)

        # Update only the phase
        _, phase = torch.abs(recomplex_spectrogram), torch.angle(recomplex_spectrogram)
        complex_spectrogram = magnitude_spectrogram * torch.exp(1j * phase)

    # Final iSTFT to get the time domain signal
    signal = torch.istft(complex_spectrogram, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length)
    
    return signal



if __name__ == '__main__':
    """How to use the Function"""

    # Load an audio TODO: set your file here
    fs, X = wavfile.read('2022-05-03T124144.208Z_r000005927_t000006595.wav')

    # Convert the signal to a PyTorch tensor
    X_tensor = torch.from_numpy(X.astype(np.float32))

    window_length_ms = 30
    window_length = int(window_length_ms * fs / 2000) * 2
    hop_length = window_length // 2
    win_length = window_length

    # Compute the magnitude spectrogram using STFT with the window length and hop length
    spectrogram_tensor = torch.stft(
        X_tensor, 
        n_fft=window_length, 
        hop_length=hop_length, 
        win_length=win_length,
        window=torch.hann_window(window_length),
        return_complex=True
    )

    # Compute the magnitude of the spectrogram
    magnitude_spectrogram = torch.abs(spectrogram_tensor)

    # Call the Griffin-Lim algorithm
    reconstructed_audio = griffin_lim(
        magnitude_spectrogram, 
        iterations=30, 
        n_fft=window_length, 
        hop_length=hop_length, 
        win_length=win_length
    )

    # Write the audio data to a WAV file
    reconstructed_audio_np = reconstructed_audio.numpy()
    reconstructed_audio_np = np.int16(reconstructed_audio_np / np.max(np.abs(reconstructed_audio_np)) * 32767)
    wavfile.write("reconstructed_spectrogram.wav", fs, reconstructed_audio_np)