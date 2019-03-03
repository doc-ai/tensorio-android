package ai.doc.tensorio.TIOLayerInterface;

import java.nio.ByteBuffer;

/**
 * Describes an input or output layer. Used internally by a model when parsing its description.
 * <p>
 * A layer description encapsulates information about an input or output tensor that is needed
 * to copy bytes into and out of it. For example, a vector layer description
 * for an input tensor describes any transformations the submitted data must undergo before the
 * underlying bytes are copied to the tensor, e.g. quantization and normalization, as well as the
 * shape of the expected input, which determines how many bytes are copied into the tensor.
 */
public abstract class TIOLayerDescription {
    /**
     * true if this data is quantized (bytes of type byte), false if not (bytes of type float)
     */
    protected boolean quantized;

    public boolean isQuantized() {
        return quantized;
    }


    public abstract ByteBuffer toByteBuffer(Object o);
    public abstract Object fromByteBuffer(ByteBuffer buffer);
    public abstract ByteBuffer getBackingByteBuffer();

}
