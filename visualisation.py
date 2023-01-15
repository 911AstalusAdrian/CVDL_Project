import librosa
import librosa.display
import matplotlib.pyplot as plt

blues = "data/genres_short/blues/blues.00000.wav"
jazz = "data/genres_short/jazz/jazz.00000.wav"
reggae = "data/genres_short/reggae/reggae.00000.wav"
blues_sig, srb = librosa.load(blues, sr=22050)
jazz_sig, srj = librosa.load(jazz, sr=22050)
reggae_sig, srr = librosa.load(reggae, sr=22050)


# Displaying the waveforms of three different genres
def plot_waveform():
    fig, ax = plt.subplots(3)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=1)

    librosa.display.waveshow(blues_sig, srb, ax=ax[0])
    ax[0].title.set_text("Blues")
    librosa.display.waveshow(reggae_sig, srr, ax=ax[1])
    ax[1].title.set_text("Reggae")
    librosa.display.waveshow(jazz_sig, srj, ax=ax[2])
    ax[2].title.set_text("Jazz")
    plt.show()


def plot_mfcc():
    blues_mfcc = librosa.feature.mfcc(blues_sig, n_fft=2048, hop_length=512, n_mfcc=15)
    librosa.display.specshow(blues_mfcc, sr=srb, hop_length=512)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("Blues MFCC")

    # show plots
    plt.show()


def plot_one_waveform():
    librosa.display.waveshow(blues_sig, srb)
    plt.title("Blues Waveform")
    plt.show()
