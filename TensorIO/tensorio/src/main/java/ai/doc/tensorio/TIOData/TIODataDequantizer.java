/*
 * TIODataDequantizer.java
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

package ai.doc.tensorio.TIOData;

/**
 * A `TIODataDequantizer` dequantizes quantized values, converting them from
 * int representations to floating point representations.
 */

public abstract class TIODataDequantizer {

    /**
     * @param value The int value that will be dequantized
     * @return A floating point representation of the value
     */

    public abstract float dequantize(int value);

    /**
     * A TIODataDequantizer that applies the provided scale and bias according to the following forumla
     * <pre>dequantized_value = (value * scale) + bias</pre>
     *
     * @param scale The scale
     * @param bias  The bias value
     * @return TIODataQuantizer
     *
     */

    public static TIODataDequantizer TIODataDequantizerWithDequantization(float scale, float bias) {
        return new TIODataDequantizer() {
            @Override
            public float dequantize(int value) {
                return (value * scale) + bias;
            }
        };
    }

    /**
     * A standard TIODataDequantizer that converts values from a range of `[0,255]` to `[0,1]`.
     * <p>
     * This is equivalent to applying a scaling factor of `1.0/255.0` and no bias.
     */

    public static TIODataDequantizer TIODataDequantizerZeroToOne() {
        float scale = 1.0f / 255.0f;
        return TIODataDequantizerWithDequantization(scale, 0f);
    }

    /**
     * A standard TIODataDequantizer that converts values from a range of `[0,255]` to `[-1,1]`.
     * <p>
     * This is equivalent to applying a scaling factor of `2.0/255.0` and a bias of `-1`.
     */

    public static TIODataDequantizer TIODataDequantizerNegativeOneToOne() {
        float scale = 2.0f / 255.0f;
        float bias = -1f;
        return TIODataDequantizerWithDequantization(scale, bias);
    }

}
