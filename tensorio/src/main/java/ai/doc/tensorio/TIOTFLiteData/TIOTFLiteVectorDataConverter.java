/*
 * TIOTFLiteVectorDataConverter.java
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

package ai.doc.tensorio.TIOTFLiteData;

import androidx.annotation.Nullable;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import ai.doc.tensorio.TIOData.TIODataConverter;
import ai.doc.tensorio.TIOData.TIODataDequantizer;
import ai.doc.tensorio.TIOData.TIODataQuantizer;
import ai.doc.tensorio.TIOLayerInterface.TIOLayerDescription;
import ai.doc.tensorio.TIOLayerInterface.TIOVectorLayerDescription;

public class TIOTFLiteVectorDataConverter implements TIODataConverter, TIOTFLiteDataConverter {

    @Override
    public ByteBuffer createBackingBuffer(TIOLayerDescription description) {
        ByteBuffer buffer;

        boolean quantized = ((TIOVectorLayerDescription)description).isQuantized();
        int length = ((TIOVectorLayerDescription)description).getLength();

        if (quantized) {
            // Layer expects bytes
            buffer = ByteBuffer.allocateDirect(length);
        } else {
            // Layer expects floats
            buffer = ByteBuffer.allocateDirect(length*4);
        }

        buffer.order(ByteOrder.nativeOrder());

        return buffer;
    }

    @Override
    public ByteBuffer toByteBuffer(Object o, TIOLayerDescription description, @Nullable ByteBuffer cache) {
        if ( !((o instanceof byte[]) || (o instanceof float[])) ) {
            throw TIOTFLiteVectorDataConverter.BadInputException();
        }

        // Create a buffer if no reusable cache is provided

        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer((TIOVectorLayerDescription)description);
        buffer.rewind();

        // Acquire needed properties from layer description

        TIOVectorLayerDescription vectorLayerDescription = (TIOVectorLayerDescription) description;
        TIODataQuantizer quantizer = vectorLayerDescription.getQuantizer();
        boolean quantized = vectorLayerDescription.isQuantized();
        int length = vectorLayerDescription.getLength();

        // Fork on float[] and bytes[]

        if (o instanceof float[]) {
            float[] floats = (float[]) o;

            if (floats.length != length) {
                throw TIOTFLiteVectorDataConverter.BadLengthException(floats.length, length);
            }

            // Fork on quantized

            if (quantized && quantizer == null) {
                throw TIOTFLiteVectorDataConverter.MissingQuantizeException();
            } else if (quantized) {
                for (float v: floats) {
                    buffer.put((byte)quantizer.quantize(v));
                }
            } else {
                FloatBuffer f = buffer.asFloatBuffer();
                f.put(floats);
            }
        }
        else if (o instanceof byte[]) {
            byte[] bytes = (byte[])o;

            if (bytes.length != length) {
                throw TIOTFLiteVectorDataConverter.BadLengthException(bytes.length, length);
            }

            buffer.put(bytes);
        }

        return buffer;
    }

    // Note that bytes are signed in java so when we read outputs from a ByteBuffer as bytes we
    // might get negative values. So we first have to unsign the byte with & 0xFF and then cast to int.

    @Override
    public Object fromByteBuffer(ByteBuffer buffer, TIOLayerDescription description) {

        // Acquire needed properties from layer description

        TIOVectorLayerDescription vectorLayerDescription = (TIOVectorLayerDescription) description;
        TIODataDequantizer dequantizer = vectorLayerDescription.getDequantizer();
        boolean quantized = vectorLayerDescription.isQuantized();
        int length = vectorLayerDescription.getLength();

        // Prepare buffer for reading

        buffer.rewind();

        // Fork on quantized

        if (quantized && dequantizer == null) {
            return buffer.array();
        } else if (quantized) {
            float[] result = new float[length];
            for (int i = 0; i < length; i++) {
                result[i] = dequantizer.dequantize((int) ( buffer.get() & 0xFF ) );
            }
            return result;
        } else {
            float[] result = new float[length];
            buffer.asFloatBuffer().get(result);
            return result;
        }
    }

    //region Exceptions

    private static IllegalArgumentException BadInputException() {
        return new IllegalArgumentException("Expected float[] or byte[] as input to the converter");
    }

    private static IllegalArgumentException BadLengthException(int given, int expected) {
        return new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected " + expected + " input has length " + given);
    }

    private static IllegalArgumentException MissingQuantizeException() {
        return new IllegalArgumentException("Float[] given as input to quantized model without quantizer, expected byte[] or quantizer != null");
    }

    //endRegion
}
