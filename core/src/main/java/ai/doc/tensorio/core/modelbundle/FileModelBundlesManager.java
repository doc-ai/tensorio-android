package ai.doc.tensorio.core.modelbundle;

import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.HashMap;

import androidx.annotation.NonNull;

public class FileModelBundlesManager extends ModelBundlesManager {

    /**
     * A File corresponding to the directory which contains the model bundles
     */

    @NonNull private File file;

    /**
     * Loads the available models in the directory specified by the file, e.g. folders that end in
     * .tiobundle or the now deprecated .tfbundle, and assigns them to the models property. Models
     * will be sorted by name by default.
     *
     * @param file The directory containing the model bundles. Only a shallow search is performed.
     * @throws IOException
     */

    public FileModelBundlesManager(@NonNull File file) throws IOException {
        if (!file.isDirectory()) {
            throw new FileNotFoundException("Not a directory");
        }

        this.file = file;


        loadFiles();
    }

    private void loadFiles() throws IOException {
        modelBundles = new HashMap<>();

        FilenameFilter filter = (dir, name) -> name.endsWith(ModelBundle.TIO_BUNDLE_EXTENSION) || name.endsWith(ModelBundle.TF_BUNDLE_EXTENSION);
        File[] contents = file.listFiles(filter);

        for (File f : contents) {
            try {
                ModelBundle bundle = new FileModelBundle(f);
                modelBundles.put(bundle.getIdentifier(), bundle);
            } catch (ModelBundle.ModelBundleException e) {
                Log.i("ModelBundleManager", "Invalid bundle: " + f.getPath());
                e.printStackTrace();
            }
        }
    }

    public void reload() {
        try {
            loadFiles();
        } catch (IOException e) {
            // This should never happen, initialization would have already caught the exception
            Log.e("ModelBundleManager", "Unexpected IO Exception loading assets: " + e.getMessage());
        }
    }
}
