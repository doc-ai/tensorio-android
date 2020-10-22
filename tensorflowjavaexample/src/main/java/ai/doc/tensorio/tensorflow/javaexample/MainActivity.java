package ai.doc.tensorio.tensorflow.javaexample;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Objects;

import ai.doc.tensorio.core.model.Model;
import ai.doc.tensorio.core.modelbundle.ModelBundle;
import ai.doc.tensorio.core.utilities.AndroidAssets;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {

            // Prepare Model

            ModelBundle bundle = bundleForFile("cats-vs-dogs-predict.tiobundle");
            Model model = bundle.newModel();

            // Load the Test Image

            InputStream stream = getAssets().open("cat.jpg");
            Bitmap bitmap = BitmapFactory.decodeStream(stream);

            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageBitmap(bitmap);

            stream.close();

            // Run Model

            Map<String,Object> output = model.runOn(bitmap);

            // Check Output

            float sigmoid = ((float[]) Objects.requireNonNull(output.get("sigmoid")))[0];

            TextView textView = findViewById(R.id.textView);
            textView.setText(formattedResults(sigmoid));

        } catch (IOException | ModelBundle.ModelBundleException | Model.ModelException e) {
            e.printStackTrace();
        }
    }

    /** Create a model bundle from a file, copying the asset to models */

    private ModelBundle bundleForFile(String filename) throws IOException, ModelBundle.ModelBundleException {
        File dir = new File(getApplicationContext().getFilesDir(), "models");
        File file = new File(dir, filename);

        if (!dir.exists()) {
            dir.mkdir();
        }

        if (!file.exists()) {
            AndroidAssets.copyAsset(getApplicationContext(), filename, file);
        }

        return ModelBundle.bundleWithFile(file);
    }

    private String formattedResults(float sigmoid) {
        StringBuilder b = new StringBuilder();

        b.append("Sigmoid output: ");
        b.append(sigmoid);
        b.append("\n");

        if (sigmoid < 0.5) {
            b.append("It's a cat!");
        } else {
            b.append("It's a dog!");
        }

        return b.toString();
    }
}