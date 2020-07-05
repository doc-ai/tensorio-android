package ai.doc.tensorio.TIOTensorflowLiteModel;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.experimental.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerDescription;
import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;
import ai.doc.tensorio.TIOModel.TIOModel;
import ai.doc.tensorio.TIOModel.TIOModelBundle;
import ai.doc.tensorio.TIOModel.TIOModelException;

public class TIOTFLiteModel extends TIOModel {

    private Interpreter tflite;
    private MappedByteBuffer tfliteModel;
    private GpuDelegate gpuDelegate = null;

    private int numThreads = 1;
    private boolean useGPU = false;
    private boolean useNNAPI = false;
    private boolean use16bit = false;

    public TIOTFLiteModel(Context context, TIOModelBundle bundle) {
        super(context, bundle);
    }

    //region Lifecycle

    @Override
    public void load() throws TIOModelException {
        try {
            tfliteModel = loadModelFile(getContext(), getBundle().getModelFilePath());
        } catch (IOException e) {
            throw new TIOModelException("Error loading model file", e);
        }
        tflite = new Interpreter(tfliteModel);
        super.load();
    }

    @Override
    public void unload() {
        if (tflite != null) {
            tflite.close();
            this.tflite = null;
        }
        if (this.gpuDelegate != null){
            this.gpuDelegate.close();
            this.gpuDelegate = null;
        }
        super.unload();
    }

    //endRegion

    //region Run

    private Map<String, Object> runMultipleInputMultipleOutput(Map input) throws TIOModelException {
        super.runOn(input);

        Object[] inputs = new Object[getIO().getInputs().size()];

        // Fetch the input and output layer descriptions from the model
        List<TIOLayerInterface> inputLayers = getIO().getInputs().all();
        List<TIOLayerInterface> outputLayers = getIO().getOutputs().all();

        for (int i=0; i<inputLayers.size(); i++){
            // Ask the input layer to parse the input object into a Bytebuffer
            TIOLayerInterface layer = inputLayers.get(i);
            ByteBuffer inputBuffer = layer.getDataDescription().toByteBuffer(input.get(layer.getName()));
            inputs[i] = inputBuffer;
        }

        Map<Integer, Object> outputs = new HashMap<>(getIO().getOutputs().size());
        for (int i=0; i<outputLayers.size(); i++){
            // Ask the output layer for a buffer to store the output in
            TIOLayerInterface layer = outputLayers.get(i);
            ByteBuffer outputBuffer = layer.getDataDescription().getBackingByteBuffer();
            outputBuffer.rewind();
            outputs.put(i, outputBuffer);
        }

        // Run the model on the input buffers, store the output in the outputbuffers
        tflite.runForMultipleInputsOutputs(inputs, outputs);

        // Ask the outputlayer to convert the outputbuffer back to an object to return to the user
        Map<String, Object> outputMap = new HashMap<>(outputLayers.size());
        for (int i=0; i<outputLayers.size(); i++){
            TIOLayerInterface layer = outputLayers.get(i);
            Object o = layer.getDataDescription().fromByteBuffer((ByteBuffer)outputs.get(i));
            outputMap.put(layer.getName(), o);
        }
        return outputMap;
    }

    private Object runSingleInputSingleOutput(Object input) throws TIOModelException {
        super.runOn(input);

        // Fetch the input and output layer descriptions from the model
        TIOLayerDescription inputLayer = getIO().getInputs().get(0).getDataDescription();
        TIOLayerDescription outputLayer = getIO().getOutputs().get(0).getDataDescription();

        // Ask the input layer to parse the input object into a Bytebuffer
        ByteBuffer inputBuffer = inputLayer.toByteBuffer(input);

        // Ask the output layer for a buffer to store the output in
        ByteBuffer outputBuffer = outputLayer.getBackingByteBuffer();
        outputBuffer.rewind();

        // Run the model on the input buffer, store the output in the outputbuffer
        tflite.run(inputBuffer, outputBuffer);

        // Ask the outputlayer to convert the outputbuffer back to an object to return to the user
        return outputLayer.fromByteBuffer(outputBuffer);
    }

    @Override
    public Object runOn(Object input) throws TIOModelException{
        super.runOn(input);

        int numInputs = getIO().getInputs().size();
        int numOutputs = getIO().getOutputs().size();

        if (numInputs > 1){
            Map<String, Object>  output = runMultipleInputMultipleOutput((Map<String, Object>)input);

            if (numOutputs == 1){
                return output.values().iterator().next();
            }
            return output;
        }
        else{
            if (input instanceof Map){
                Map<String, Object>  output = runMultipleInputMultipleOutput((Map<String, Object>)input);
                if (numOutputs == 1){
                    return output.values().iterator().next();
                }
                return output;
            }
            else{
                if (numOutputs == 1) {
                    return runSingleInputSingleOutput(input);
                }
                else{
                    Map<String, Object> inputMap = new HashMap<>();
                    TIOLayerInterface inputLayer = getIO().getInputs().get(0);
                    inputMap.put(inputLayer.getName(), input);
                    return runMultipleInputMultipleOutput(inputMap);
                }
            }
        }
    }

    //endRegion

    //region Utilities

    private MappedByteBuffer loadModelFile(Context context, String path) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(path);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void recreateInterpreter() {
        unload();
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setAllowFp16PrecisionForFp32(use16bit);
        tfliteOptions.setUseNNAPI(useNNAPI);
        tfliteOptions.setNumThreads(numThreads);
        if (useGPU && GpuDelegateHelper.isGpuDelegateAvailable()){
            tfliteOptions.addDelegate((GpuDelegate)GpuDelegateHelper.createGpuDelegate());
        }

        tflite = new Interpreter(tfliteModel, tfliteOptions);

    }

    public void useGPU() {
        if (GpuDelegateHelper.isGpuDelegateAvailable()){
            useGPU = true;
            recreateInterpreter();
        }
    }

    public void useCPU() {
        useGPU = false;
        useNNAPI = false;
        recreateInterpreter();
    }

    public void useNNAPI() {
        useGPU = false;
        useNNAPI = true;
        recreateInterpreter();
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
        recreateInterpreter();
    }

    public void setAllow16BitPrecision(boolean use16Bit) {
        this.use16bit = use16Bit;
        recreateInterpreter();
    }

    public long getLastInferenceDuration(){
        return tflite.getLastNativeInferenceDurationNanoseconds();
    }

    public void setOptions(boolean use16bit, boolean useGPU, boolean useNNAPI, int numThreads){
        this.use16bit = use16bit;
        if (useGPU && GpuDelegateHelper.isGpuDelegateAvailable()){
            this.useGPU = true;
            this.useNNAPI = false;
        }
        else if (useNNAPI){
            this.useGPU = false;
            this.useNNAPI = true;
        }
        this.numThreads = numThreads;
        recreateInterpreter();
    }

    //endRegion
}
