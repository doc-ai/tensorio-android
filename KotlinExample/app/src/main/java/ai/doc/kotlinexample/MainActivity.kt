package ai.doc.kotlinexample

import ai.doc.tensorio.TIOModel.TIOModelBundle
import ai.doc.tensorio.TIOModel.TIOModelBundleException
import ai.doc.tensorio.TIOModel.TIOModelException
import ai.doc.tensorio.TIOUtilities.TIOClassificationHelper
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import java.io.IOException

private const val TAG = "MainActivity"

class MainActivity : AppCompatActivity() {

    private val main by lazy { Handler(Looper.getMainLooper()) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            // Load the Model

            val bundle = TIOModelBundle(applicationContext, "mobilenet_v2_1.4_224.tiobundle")
            val model = bundle.newModel()

            // Load the Image

            val stream = assets.open("picture2.jpg")
            val bitmap = BitmapFactory.decodeStream(stream)
            val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, false)

            val imageView = findViewById<ImageView>(R.id.imageView)
            imageView.setImageBitmap(bitmap)

            stream.close()

            // Create a Background Thread

            val mHandlerThread = HandlerThread("HandlerThread")
            mHandlerThread.start()
            val mHandler = Handler(mHandlerThread.looper)

            // Execute the Model

            mHandler.post {
                try {
                    val output = model.runOn(scaled)
                    val classification = output.get("classification") as MutableMap<String, Float>
                    val top5 = TIOClassificationHelper.topN(classification, 5, 0.1f)

                    for (entry in top5) {
                        Log.i(TAG, entry.key + ":" + entry.value)
                    }

                    main.post {
                        val textView = findViewById<TextView>(R.id.textView)
                        textView.text = formattedResults(top5)
                    }

                } catch (e: TIOModelException) {
                    e.printStackTrace()
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        } catch (e: TIOModelBundleException) {
            e.printStackTrace()
        }
    }

    private fun formattedResults(results: List<Map.Entry<String, Float>>): String? {
        val b = StringBuilder()

        for ((key, value) in results) {
            b.append(key)
            b.append(": ")
            b.append(value)
            b.append("\n")
        }

        b.setLength(b.length - 1)

        return b.toString()
    }
}
