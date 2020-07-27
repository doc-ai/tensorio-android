package ai.doc.tensorio.core.model;

import androidx.annotation.NonNull;

public class TIOModelBundleValidatorException extends Exception {
    public TIOModelBundleValidatorException(@NonNull String message) {
        super(message);
    }

    public TIOModelBundleValidatorException(@NonNull String message, @NonNull Throwable cause) {
        super(message, cause);
    }
}
