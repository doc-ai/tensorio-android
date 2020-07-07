package ai.doc.kotlinexample

import ai.doc.tensorio.TIOModel.TIOModelBundleException
import ai.doc.tensorio.TIOModel.TIOModelBundleManager
import ai.doc.tensorio.TIOModel.TIOModelException
import ai.doc.tensorio.TIOUtilities.TIOClassificationHelper
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import java.io.IOException
import java.util.*

private const val TAG = "MainActivity"

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            val manager = TIOModelBundleManager(applicationContext, "")

            // load the model
            val bundle = manager.bundleWithId("mobilenet-v2-100-224-unquantized")
            val model = bundle.newModel()
            model.load()

            // Load the image
            val bitmap = assets.open("picture2.jpg")
            val bMap = BitmapFactory.decodeStream(bitmap)
            val scaled = Bitmap.createScaledBitmap(bMap, 224, 224, false)

            // Create a background thread
            val mHandlerThread = HandlerThread("HandlerThread")
            mHandlerThread.start()
            val mHandler = Handler(mHandlerThread.looper)

            mHandler.post {
                try {
                    val output = model.runOn(scaled)
                    val classification = output.get("classification") as MutableMap<String, Float>
                    val top5 = TIOClassificationHelper.topN(classification, 5)

                    for (entry in top5) {
                        Log.i(TAG, entry.key + ":" + entry.value)
                    }
                } catch (e: TIOModelException) {
                    e.printStackTrace()
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        } catch (e: TIOModelBundleException) {
            e.printStackTrace()
        } catch (e: TIOModelException) {
            e.printStackTrace()
        }

    }
}
