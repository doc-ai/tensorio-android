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

import android.support.annotation.Nullable;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import ai.doc.tensorio.TIOData.TIODataConverter;
import ai.doc.tensorio.TIOData.TIODataDequantizer;
import ai.doc.tensorio.TIOData.TIODataQuantizer;
import ai.doc.tensorio.TIOLayerInterface.TIOLayerDescription;
import ai.doc.tensorio.TIOLayerInterface.TIOVectorLayerDescription;
import ai.doc.tensorio.TIOTFLiteData.TIOTFLiteDataConverter;

public class TIOTFLiteVectorDataConverter implements TIODataConverter, TIOTFLiteDataConverter {

    /**
     * Backing buffer
     */

    private ByteBuffer _buffer;

    public TIOTFLiteVectorDataConverter() {}

    public TIOTFLiteVectorDataConverter(TIOVectorLayerDescription description) {
        _buffer = createBackingBuffer(description);
    }

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

    // TODO: Where is the quantizer being applied? (#28)

    @Override
    public ByteBuffer toByteBuffer(Object o, TIOLayerDescription description, @Nullable ByteBuffer cache) {
        ByteBuffer buffer = (cache != null) ? cache : createBackingBuffer((TIOVectorLayerDescription)description);

        TIOVectorLayerDescription vectorLayerDescription = (TIOVectorLayerDescription) description;
        TIODataQuantizer quantizer = vectorLayerDescription.getQuantizer();
        boolean quantized = vectorLayerDescription.isQuantized();
        int length = vectorLayerDescription.getLength();

        buffer.rewind();

        if (o instanceof float[]){
            if (quantized){
                if (quantizer != null){
                    FloatBuffer f = buffer.asFloatBuffer();
                    float[] floatInput = (float[])o;
                    if (floatInput.length != length){
                        throw new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected "+length+" input has length "+floatInput.length);
                    }
                    for (float v: floatInput){
                        f.put(v);
                    }
                }
                else{
                    throw new IllegalArgumentException("Float[] given as input to quantized model without quantizer, expected byte[] or quantizer");
                }
            }
            else{
                float[] floatInput = (float[])o;
                if (floatInput.length != length){
                    throw new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected "+length+" input has length "+floatInput.length);
                }
                FloatBuffer f = buffer.asFloatBuffer();
                f.put(floatInput);
            }
        }
        else if (o instanceof byte[]){
            byte[] byteInput = (byte[])o;
            if (byteInput.length != length){
                throw new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected "+length+" input has length "+byteInput.length);
            }
            buffer.put(byteInput);
        }
        else{
            throw new IllegalArgumentException("Expected float[] or byte[] as input to the model");
        }

        return buffer;
    }

    /**
     * Note that bytes are signed in java so when we read outputs from a ByetBuffer as bytes we
     * might get negative values. So we first have to unsign the byte with & 0xFF and then cast to int.
     * Jesus.
     * @param byteBuffer
     * @return
     */

    @Override
    public Object fromByteBuffer(ByteBuffer byteBuffer, TIOLayerDescription description) {
        TIOVectorLayerDescription vectorLayerDescription = (TIOVectorLayerDescription) description;
        TIODataDequantizer dequantizer = vectorLayerDescription.getDequantizer();
        boolean quantized = vectorLayerDescription.isQuantized();
        int length = vectorLayerDescription.getLength();

        if (quantized){
            if (dequantizer != null){
                float[] result = new float[length];
                byteBuffer.rewind();

                for (int i=0; i<length; i++) {
                    result[i] = dequantizer.dequantize((int) ( byteBuffer.get() & 0xFF ) );
                }
                return result;
            }
            else {
                return byteBuffer.array();
                //int[] result = new int[this.length];
                //byteBuffer.rewind();
                //byteBuffer.asIntBuffer().get(result);
                //byteBuffer.asIntBuffer().get(result);
                //return result;
            }
        }
        else {
            float[] result = new float[length];
            byteBuffer.rewind();
            byteBuffer.asFloatBuffer().get(result);
            return result;
        }
    }

    @Override
    public ByteBuffer getBackingByteBuffer() {
        return _buffer;
    }
}
