package ai.doc.tensorio.core.utilities;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

import androidx.annotation.NonNull;

public class AndroidAssets {

    /**
     * Reads a UTF_8 text file in the assets directory and returns its String contents
     *
     * @param c The context providing the assets
     * @param filename The filename of the asset, may include directories
     * @return The String contents of the file
     * @throws IOException when any part of the read operation goes wrong
     */

    public static String readTextFile(@NonNull Context c, @NonNull String filename) throws IOException {
        InputStream is = c.getAssets().open(filename);
        int size = is.available();
        byte[] buffer = new byte[size];
        is.read(buffer);
        is.close();
        return new String(buffer, StandardCharsets.UTF_8);
    }

    /**
     * Copies the asset named filename to the target destination, creating a file or directory
     * at that location
     *
     * @param context The context containing the asset to copy
     * @param filename The name of the asset to copy, a file or directory
     * @param destination The target destination file to copy the asset to
     * @throws IOException when any part of the copy operation goes wrong
     */

    // Replace String destination with File destination

    public static void copyAsset(@NonNull Context context, @NonNull String filename, @NonNull File destination) throws IOException {
        copyFileOrDir(context.getAssets(), filename, destination);
    }

    private static void copyFileOrDir(AssetManager assetManager, String path, File destination) throws IOException {
        String assets[] = assetManager.list(path);

        if ( assets == null ) {
            throw new FileNotFoundException();
        }

        if ( assets.length == 0 ) {
            copyFile(assetManager, path, destination);
        } else {
            if (!destination.exists())
                destination.mkdir();

            for (String asset : assets) {
                String targetPath = path + "/" + asset;
                File targetDestination = new File(destination, asset);
                copyFileOrDir(assetManager, targetPath, targetDestination);
            }
        }
    }

    private static void copyFile(AssetManager assetManager, String filename, File destination) throws IOException {
        InputStream in = assetManager.open(filename);
        OutputStream out = new FileOutputStream(destination);

        byte[] buffer = new byte[1024];
        int read;

        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }

        in.close();
        in = null;

        out.flush();
        out.close();
        out = null;
    }
}
