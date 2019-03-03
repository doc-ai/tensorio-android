package ai.doc.tensorio.TIOLayerInterface;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.doc.tensorio.TIOData.TIODataDequantizer;
import ai.doc.tensorio.TIOData.TIODataQuantizer;

/**
 * The description of a vector (array) input or output later.
 * <p>
 * Vector inputs and outputs are always unrolled vectors, and from the tensor's perspective they are
 * just an array of bytes. The total length of a vector will be the total volume of the layer.
 * For example, if an input layer is a tensor of shape `(24,24,2)`, the length of the vector will be
 * `24x24x2 = 1152`.
 * <p>
 * TensorFlow Lite models expect row major ordering of bytes, such that higher order dimensions are
 * traversed first. For example, a 2x4 matrix with the following values:
 *
 * <pre>
 * [[1 2 3 4]
 * [5 6 7 8]]
 * </pre>
 * <p>
 * <p>
 * should be unrolled and provided to the model as:
 *
 * <pre>
 * [1 2 3 4 5 6 7 8]
 * </pre>
 * <p>
 * i.e, start with the row and traverse the columns before moving to the next row.
 * <p>
 * Because output layers are also exposed as an array of bytes, a `TIOTFLiteModel` will always return
 * a vector in one dimension. If is up to you to reshape it if required.
 * <p>
 */

public class TIOVectorLayerDescription extends TIOLayerDescription {

    private final int[] shape;
    private ByteBuffer buffer;

    /**
     * The length of the vector in terms of its number of elements.
     */

    private int length;

    /**
     * Indexed labels corresponding to the indexed output of a layer. May be `nil`.
     * <p>
     * Labeling the output of a model is such a common operation that support for it is included
     * by default.
     */

    private String[] labels;

    /**
     * `YES` if there are labels associated with this layer, `NO` otherwise.
     */

    private boolean labeled;

    /**
     * A function that converts a vector from unquantized values to quantized values
     */

    private TIODataQuantizer quantizer;

    /**
     * A function that converts a vector from quantized values to unquantized values
     */

    private TIODataDequantizer dequantizer;

    /**
     * Designated initializer. Creates a vector description from the properties parsed in a model.json
     * file.
     *
     * @param labels      The indexed labels associated with the outputs of this layer. May be `nil`.
     * @param quantized   True if the values are quantized
     * @param quantizer   A function that transforms unquantized values to quantized input
     * @param dequantizer A function that transforms quantized output to unquantized values
     */

    public TIOVectorLayerDescription(int[] shape, String[] labels, boolean quantized, TIODataQuantizer quantizer, TIODataDequantizer dequantizer) {
        this.shape = shape;

        // Total Volume
        this.length = 1;
        for (int i : shape) {
            length *= i;
        }

        this.labels = labels;
        this.labeled = labels != null && labels.length > 0;
        this.quantized = quantized;
        this.quantizer = quantizer;
        this.dequantizer = dequantizer;

        if (quantized){
            this.buffer = ByteBuffer.allocate(length);
        }
        else{
            this.buffer = ByteBuffer.allocate(length*4);
        }
        this.buffer.order(ByteOrder.nativeOrder());

    }

    public int getLength() {
        return length;
    }

    public String[] getLabels() {
        return labels;
    }

    public boolean isLabeled() {
        return labeled;
    }

    public TIODataQuantizer getQuantizer() {
        return quantizer;
    }

    public TIODataDequantizer getDequantizer() {
        return dequantizer;
    }

    @Override
    public ByteBuffer toByteBuffer(Object o) {
        buffer.rewind();
        if (o instanceof float[]){
            if (quantized){
                if (quantizer != null){
                    FloatBuffer f = buffer.asFloatBuffer();
                    float[] floatInput = (float[])o;
                    if (floatInput.length != this.length){
                        throw new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected "+this.length+" input has length "+floatInput.length);
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
                if (floatInput.length != this.length){
                    throw new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected "+this.length+" input has length "+floatInput.length);
                }
                FloatBuffer f = buffer.asFloatBuffer();
                f.put(floatInput);
            }
        }
        else if (o instanceof byte[]){
            byte[] byteInput = (byte[])o;
            if (byteInput.length != this.length){
                throw new IllegalArgumentException("Provided input is of different size than the size expected by the model, expected "+this.length+" input has length "+byteInput.length);
            }
            buffer.put(byteInput);
        }
        else{
            throw new IllegalArgumentException("Expected float[] or byte[] as input to the model");
        }

        return buffer;
    }

    @Override
    public Object fromByteBuffer(ByteBuffer buffer) {
        if (quantized){
            if (dequantizer != null){
                float[] result = new float[this.length*4];
                buffer.rewind();
                for (int i=0; i<this.length; i++) {
                    result[i] = dequantizer.dequantize(buffer.get());
                }
                return result;
            }
            else{
                return buffer.array();
                //int[] result = new int[this.length];
                //buffer.rewind();
                //buffer.asIntBuffer().get(result);
                //buffer.asIntBuffer().get(result);
                //return result;
            }
        }
        else{
            float[] result = new float[this.length];
            buffer.rewind();
            buffer.asFloatBuffer().get(result);
            return result;
        }
    }

    @Override
    public ByteBuffer getBackingByteBuffer() {
        return buffer;
    }

    /**
     * Given the output vector of a tensor, returns labeled outputs using `labels`.
     *
     * @param vector An array of float values.
     * @return  The labeled values, where the dictionary keys are the labels and the
     * dictionary values are the associated vector values.
     */

    public Map<String, Float> labeledValues(float[] vector) {
        if (!isLabeled()){
            return null;
        }
        Map<String, Float> result = new HashMap<>(vector.length);
        for (int i = 0; i < labels.length; i++){
            result.put(labels[i], vector[i]);
        }
        return result;
    }


}
