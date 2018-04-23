import soundfile as sf
from scipy.io import wavfile
import os
from os.path import basename
import cv2
import numpy as np
seed = 0
np.random.seed(seed)
import random
random.seed(seed)

def randomMix(X, sample_rate, speaker, command, u=0.5):
    if np.random.random() < u:
        mix_folders = os.listdir('./Data/train/audio/')
        mix_folders.remove('_background_noise_')
        mix_folders.remove(command)
        mix_folders.remove('house')
        mix_folders.remove('dog')
        mix_folders.remove('two')
        mix_folders.remove('five')
        # Get same speaker's file from some other folder
        mix2 = None
        trial = 0
        while mix2 is None:
            try:
                chosen_folder = np.random.choice(mix_folders)
                mix2, sr = sf.read('./Data/train/audio/'+chosen_folder+'/'+speaker+'_nohash_0.wav')
                mix2, _ = librosa.effects.trim(mix2)
            except:
                trial+=1
                if trial>29:
                    return X
                pass
        len_mix1 = np.random.randint(len(X)//3, len(X)//2)
        start_mix1 = np.random.randint(len(X)-len_mix1)
        mix1_slice = X[start_mix1+len(X)//6 : start_mix1+len_mix1+len(X)//6]
        len_mix2 = np.random.randint(len(mix2)//3, len(mix2)//2)
        start_mix2 = np.random.randint(len(mix2)-len_mix2)
        mix2_slice = mix2[start_mix2+len(mix2)//6: start_mix2+len_mix2+len(mix2)//6]
        X = np.concatenate((mix1_slice, mix2_slice))
    return X


def randomNoise(X, sample_rate, bg_wavs, sr=16000, noise=(0, 0.2), u=0.5):
    if np.random.random() < u:
        bg_files = ['pink_noise.wav', 'dude_miaowing.wav', 'exercise_bike.wav',
                    'doing_the_dishes.wav', 'white_noise.wav', 'running_tap.wav']
        chosen_bg_file = np.random.choice(bg_files)
        bg = bg_wavs[chosen_bg_file]
        try:
            start_ = np.random.randint(bg.shape[0]-len(X))
            bg_slice = bg[start_ : start_+len(X)]
            X = X * np.random.uniform(0.8, 1.2) + \
                      bg_slice * np.random.uniform(noise[0], noise[1])
        except ValueError:
            pass
        
    return X


def randomVol(X, u=0.5):
    if np.random.random() < u:
        scale = np.random.randint(2, 6) 
        if np.random.randint(2):
            X = X * scale
        else:
            X = X /scale 
        
    return X


def randomSpeed(X, sample_rate, rate = (0.6, 1.4), u=0.5):
    if np.random.random() < u:
        speed_rate = np.random.uniform(rate[0], rate[1])
        wav_speed_tune = cv2.resize(X, (1, int(len(X) * speed_rate))).squeeze()
        if len(wav_speed_tune) < sample_rate:
            pad_len = sample_rate - len(wav_speed_tune)
            X = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                                   wav_speed_tune,
                                   np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
        else: 
            cut_len = len(wav_speed_tune) - sample_rate
            X = wav_speed_tune[int(cut_len/2):int(cut_len/2)+sample_rate]
    return X
