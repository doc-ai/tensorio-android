package ai.doc.tensorio.core.modelbundle;

import androidx.annotation.NonNull;

public class ValidatorException extends Exception {
    public ValidatorException(@NonNull String message) {
        super(message);
    }

    public ValidatorException(@NonNull String message, @NonNull Throwable cause) {
        super(message, cause);
    }
}
