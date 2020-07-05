package ai.doc.tensorio.TIOData;

/**
 * Describes how pixel values in the range of `[0,255]` will be normalized for
 * non-quantized, float32 models.
 * <p>
 * Pixels will typically normalized to values in the range `[0,1]` or `[-1,+1]`,
 * although separate biases may be applied to each of the RGB channels.
 * <p>
 * Pixel normalization is like quantization but in the opposite direction.
 */

public abstract class TIOPixelNormalizer {
    /**
     * A `TIOPixelNormalizer` transforms a pixel value in the range `[0,255]`
     * to some other range, where the transformation may be channel dependent.
     *
     * @param value   The single byte pixel value being transformed.
     * @param channel The RGB channel of the pixel value being transformed.
     * @return float The transformed value.
     */

    public abstract float normalize(int value, int channel);

    /**
     * A TIOPixelNormalizer that applies a scaling factor and equal bias to each pixel channel.
     */

    public static TIOPixelNormalizer TIOPixelNormalizerSingleBias(float scale, float bias) {
        return new TIOPixelNormalizer() {
            @Override
            public float normalize(int value, int channel) {
                return (value * scale) + bias;
            }
        };
    }

    /**
     * A TIOPixelNormalizer that applies a scaling factor and different biases to each pixel channel.
     */

    public static TIOPixelNormalizer TIOPixelNormalizerPerChannelBias(float scale, float redBias, float greenBias, float blueBias) {
        return new TIOPixelNormalizer() {
            @Override
            public float normalize(int value, int channel) {
                switch (channel) {
                    case 0:
                        return (value * scale) + redBias;
                    case 1:
                        return (value * scale) + greenBias;
                    default:
                        return (value * scale) + blueBias;
                }
            }
        };
    }

    /**
     * Normalizes pixel values from a range of `[0,255]` to `[0,1]`.
     * <p>
     * This is equivalent to applying a scaling factor of `1.0/255.0` and no channel bias.
     */

    public static TIOPixelNormalizer TIOPixelNormalizerZeroToOne(){
        float scale = 1.0f/255.0f;
        return TIOPixelNormalizerSingleBias(scale, 0.0f);
    }

    /**
     * Normalizes pixel values from a range of `[0,255]` to `[-1,1]`.
     * <p>
     * This is equivalent to applying a scaling factor of `2.0/255.0` and a bias of `-1` to each channel.
     */

    public static TIOPixelNormalizer TIOPixelNormalizerNegativeOneToOne(){
        float scale = 2.0f/255.0f;
        float bias = -1f;
        return TIOPixelNormalizerSingleBias(scale, bias);
    }
}
