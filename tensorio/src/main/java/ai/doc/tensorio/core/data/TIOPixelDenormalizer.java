/*
 * TIOPixelDenormalizer.java
 * TensorIO
 *
 * Created by Philip Dow on 7/6/2020
 * Copyright (c) 2020 - Present doc.ai (http://doc.ai)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.doc.tensorio.core.data;

/**
 * Describes a denormalization, or how pixel values in some arbitrary range will be
 * denormalized back to pixel values in the range of `[0,255]`
 * <p>
 * Pixels will typically be denormalized from values in the range `[0,1]` or `[-1,+1]`,
 * although separate denormaliation biases may be required for each of the RGB channels.
 * <p>
 * Normalization and denormalization apply the same operations with scaling and bias values,
 * but they are typically inverses of one another.
 */

public abstract class TIOPixelDenormalizer {
    /**
     * A `TIOPixelNormalizer` transforms a pixel value in the range `[0,255]`
     * to some other range, where the transformation may be channel dependent.
     *
     * @param value   The single byte pixel value being transformed.
     * @param channel The RGB channel of the pixel value being transformed.
     * @return The transformed value.
     */

    public abstract int denormalize(float value, int channel);

    /**
     * A TIOPixelDenormalizer that applies a scaling factor and equal bias to each pixel channel.
     */

    public static TIOPixelDenormalizer TIOPixelDenormalizerSingleBias(float scale, float bias) {
        return new TIOPixelDenormalizer() {
            @Override
            public int denormalize(float value, int channel) {
                return (int) ((value + bias) * scale);
            }
        };
    }

    /**
     * A TIOPixelDenormalizer that applies a scaling factor and different biases to each pixel channel.
     */

    public static TIOPixelDenormalizer TIOPixelDenormalizerPerChannelBias(float scale, float redBias, float greenBias, float blueBias) {
        return new TIOPixelDenormalizer() {
            @Override
            public int denormalize(float value, int channel) {
                switch (channel) {
                    case 0:
                        return (int) ((value + redBias) * scale);
                    case 1:
                        return (int) ((value + greenBias) * scale);
                    default:
                        return (int) ((value + blueBias) * scale);
                }
            }
        };
    }


    /**
     * Denormalizes pixel values from a range of `[0,1]` to `[0,255]`.
     * <p>
     * This is equivalent to applying no channel bias a scaling factor of `255.0`.
     */

    public static TIOPixelDenormalizer TIOPixelDenormalizerZeroToOne() {
        float scale = 255.0f;
        return TIOPixelDenormalizerSingleBias(scale, 0.0f);
    }

    /**
     * Denormalizes pixel values from a range of `[-1,1]` to `[0,255]`.
     * <p>
     * This is equivalent to applying a bias of `1` to each channel and a scaling factor of `255.0/2.0`.
     */

    public static TIOPixelDenormalizer TIOPixelDenormalizerNegativeOneToOne() {
        float scale = 255.0f / 2.0f;
        float bias = 1f;
        return TIOPixelDenormalizerSingleBias(scale, bias);
    }

}
