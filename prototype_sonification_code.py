import numpy as np
import wave

def sonify(in, out, maxInSec=10):
    bits = 32
    sampleRate = 48000
    # We can control oversampling level from here
    oversample = 4

    # take in some payload.
    with open(in, 'rb') as inputFile:
        inputFile.seek(44)
        inSmpsNum = (sampleRate * maxInSec) // (bits * oversample)
        data = np.fromfile(inputFile, dtype=np.float32, count=inSmpsNum)

     # //Extraction of bits
    samples = (data * 2147483647).astype(np.int32)
    mask = (1 << np.arange(bits - 1, -1, -1, dtype=np.int64)).astype(np.uint32)
    sdBits = (samples[:, None].astype(np.uint32) & mask) > 0

    # //Oversampling on the raw bits
    sdRaw = sdBits.flatten().astype(np.float32) * 2 - 1
    sdDel = np.repeat(sdRaw, oversample)

    # /produce the delay
    sdline = np.zeros_like(sdDel)
    sdline[oversample:] = sdDel[:-oversample]

    # generate the clock and keep locked to OS rate
    # Bit Clock: 1 high/low cycle per oversample
    sckchanges = [1.0] * oversample + [-1.0] * oversample
    SCK = np.tile(sckchanges, (len(sdDel) // (oversample * 2)) + 1)[:len(sdDel)]

    #  do Word Select: (again locked to oversample)
    wsChanges = np.repeat([1.0, -1.0], bits * oversample)
    wsline = np.tile(wsChanges, (len(sdDel) // (bits * oversample * 2)) + 1)[:len(sdDel)]

    # //stereo mix
    left = (0.02 * wsline) + (0.03 * SCK)
    right = sdline * 0.05

    # file output
    stereo = np.stack((left, right), axis=-1)
    pcm out = (stereo * 32767).astype(np.int16)

    with wave.open(out, 'wb') as f_out:
        f_out.setnchannels(2); f_out.setsampwidth(2); f_out.setframerate(sampleRate)
        f_out.writeframes(pcm out.tobytes())

    print(f"I2S Transport Data Sonified!! ")

sonify("input.wav", "i2s_son9.wav")
