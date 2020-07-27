package ai.doc.tensorio.tflite.model;

import android.os.Build;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

public class NnApiDelegateHelper {

    /**
     * Checks whether {@code NnApiDelegate} is available.
     */

    public static boolean isNnApiDelegateAvailable() {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.P;
    }

    /**
     * Returns an instance of {@code GpuDelegate} if available.
     */

    public static Delegate createNnApiDelegate() {
        if (!isNnApiDelegateAvailable()) {
            throw new IllegalStateException();
        } else {
            return new NnApiDelegate();
        }
    }
}
