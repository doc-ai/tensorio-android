/*
 * DataQuantizer.java
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

// TODO: Quantization should produce bytes not ints but there are is no unsigned byte primitive (#28)

package ai.doc.tensorio.core.data;

/**
 * A `DataQuantizer` quantizes unquantized values, converting them from
 * floating point representations to int representations.
 */

public abstract class Quantizer {

    /**
     * @param value The float value that will be quantized
     * @return A quantized representation of the value
     */

    public abstract int quantize(float value);

    /**
     * A DataQuantizer that applies the provided scale and bias according to the following formula:
     *
     * <pre>
     * quantized_value = (value + bias) * scale
     * </pre>
     *
     * @param scale The scale
     * @param bias  The bias values
     * @return DataQuantizer
     */

    public static Quantizer DataQuantizerWithQuantization(float scale, float bias) {
        return new Quantizer() {
            @Override
            public int quantize(float value) {
                return (int)((value+bias) * scale);
            }
        };
    }

    /**
     * A standard DataQuantizer function that converts values from a range of `[0,1]` to `[0,255]`
     */

    public static Quantizer DataQuantizerZeroToOne() {
        return new Quantizer() {
            @Override
            public int quantize(float value) {
                return (int)(value * 255.0f);
            }
        };
    }

    /**
     * A standard DataQuantizer function that converts values from a range of `[-1,1]` to `[0,255]`
     */

    public static Quantizer DataQuantizerNegativeOneToOne() {
        float scale = 255.0f/2.0f;
        float bias = 1f;
        return DataQuantizerWithQuantization(scale, bias);
    }
}
