package ai.doc.tensorio.core.modelbundle;

import android.content.Context;
import android.util.Log;

import java.io.IOException;
import java.util.HashMap;

import androidx.annotation.NonNull;

public class AssetModelBundlesManager extends ModelBundlesManager {

    /**
     * The application or activity context whose assets contain the model bundles
     */

    @NonNull Context context;

    /**
     * The path to the directory containing the model bundles
     */

    @NonNull String path;

    /**
     * Loads the available models at the relative path in context, e.g. folders that end in .tiobundle or
     * the now deprecated .tfbundle, and assigns them to the models property. Models will be sorted
     * by name by default.
     *
     * @param c The `Context` containing an assets folder and models at path
     * @param path Directory in the context's assets folder where model bundles are located. Only
     *             a shallow search is performed.
     */

    public AssetModelBundlesManager(@NonNull Context c, @NonNull String path) throws IOException {
        this.context = c;
        this.path = path;

        loadAssets();
    }

    private void loadAssets() throws IOException {
        modelBundles = new HashMap<>();

        String[] assets = context.getAssets().list(path);

        if (assets == null) {
            return;
        }

        for (String s: assets) {
            if ( !(s.endsWith(ModelBundle.TF_BUNDLE_EXTENSION) || s.endsWith(ModelBundle.TIO_BUNDLE_EXTENSION)) ) {
                continue;
            }

            try {
                ModelBundle bundle = new AssetModelBundle(context, s);
                modelBundles.put(bundle.getIdentifier(), bundle);
            } catch (ModelBundle.ModelBundleException e) {
                Log.i("ModelBundleManager", "Invalid bundle: " + s);
                e.printStackTrace();
            }
        }
    }

    public void reload() {
        try {
            loadAssets();
        } catch (IOException e) {
            // This should never happen, initialization would have already caught the exception
            Log.e("ModelBundleManager", "Unexpected IO Exception loading assets: " + e.getMessage());
        }
    }
}
