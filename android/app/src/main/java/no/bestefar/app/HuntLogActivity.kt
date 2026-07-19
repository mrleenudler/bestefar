package no.bestefar.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Geocoder
import android.location.LocationManager
import android.os.Bundle
import android.text.InputFilter
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.Space
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import java.util.Locale
import kotlin.math.cos
import kotlin.math.sin

/**
 * Jaktlogg (musingsUI-runde 2), to sider:
 *  1) posisjon (auto/fritekst/hopp over) → vilt (2 rader) → avstand → Neste
 *  2) vinkling som «Posisjon 1–6» i klokkeform rundt jaktikonet →
 *     «Dyret løp X m» / Ettersøk / Bomskudd → Avbryt / Registrer skudd.
 */
class HuntLogActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var root: FrameLayout

    // Side 1-tilstand
    private var lat: Double? = null
    private var lon: Double? = null
    private var placeName: String = ""
    private var species: Species? = null
    private var speciesOther: String = ""
    private var distanceM: Int? = null
    private var posAsked = false

    // Side 2-tilstand
    private var clockPos: Int? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        root = FrameLayout(this)
        Ui.applyInsets(root)
        setContentView(root)
        tryFetchLocation()
        page1()
    }

    // ---------- Posisjon ----------

    private fun hasLocationPermission() = ContextCompat.checkSelfPermission(this,
        Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

    @SuppressLint("MissingPermission")
    private fun tryFetchLocation() {
        if (!hasLocationPermission()) return
        val lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val loc = try {
            lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                ?: lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
        } catch (_: Exception) { null } ?: return
        lat = loc.latitude; lon = loc.longitude
        // Stedsnavn er best-effort (Geocoder trenger ofte nett)
        Thread {
            try {
                @Suppress("DEPRECATION")
                val res = Geocoder(this, Locale.getDefault())
                    .getFromLocation(loc.latitude, loc.longitude, 1)
                val name = res?.firstOrNull()?.let {
                    it.subAdminArea ?: it.locality ?: it.adminArea
                } ?: ""
                runOnUiThread {
                    if (name.isNotBlank() && placeName.isBlank()) {
                        placeName = name
                        page1()
                    }
                }
            } catch (_: Exception) { }
        }.start()
    }

    private fun askForPosition() {
        if (posAsked) return
        posAsked = true
        AlertDialog.Builder(this)
            .setTitle(R.string.hunt_position)
            .setItems(arrayOf(getString(R.string.hunt_pos_allow),
                getString(R.string.hunt_pos_manual),
                getString(R.string.hunt_pos_skip))) { _, i ->
                when (i) {
                    0 -> ActivityCompat.requestPermissions(this,
                        arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), 3)
                    1 -> manualPlaceDialog()
                }
            }
            .show()
    }

    private fun manualPlaceDialog() {
        val input = EditText(this).apply { hint = getString(R.string.hunt_position) }
        AlertDialog.Builder(this)
            .setTitle(R.string.hunt_pos_manual)
            .setView(input)
            .setPositiveButton(R.string.ok) { _, _ ->
                placeName = input.text.toString().trim()
                page1()
            }
            .show()
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>,
                                            results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == 3 && results.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            tryFetchLocation()
            page1()
        }
    }

    // ---------- Side 1 ----------

    private fun page1() {
        val content = Ui.col(this)

        val posText = when {
            placeName.isNotBlank() && lat != null ->
                "$placeName (%.4f, %.4f)".format(lat, lon)
            placeName.isNotBlank() -> placeName
            lat != null -> "%.4f, %.4f".format(lat, lon)
            else -> "—"
        }
        content.addView(Ui.body(this, "${getString(R.string.hunt_position)}: $posText"))
        if (lat == null && placeName.isBlank()) {
            if (!hasLocationPermission()) askForPosition()
            content.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = getString(R.string.hunt_pos_manual)
                setOnClickListener { manualPlaceDialog() }
            })
        }

        // Avstandsfeltet opprettes før viltknappene så innholdet kan bevares
        // når siden tegnes på nytt ved artsvalg
        val distInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            filters = arrayOf(InputFilter.LengthFilter(4))
            minWidth = Ui.dp(this@HuntLogActivity, 80)
            setText(distanceM?.toString() ?: "")
        }

        // Vilt i to rader: Rådyr, Hjort, Elg | Villrein, Villsvin, Annet
        val speciesRows = listOf(
            listOf(Species.RAADYR, Species.HJORT, Species.ELG),
            listOf(Species.VILLREIN, Species.VILLSVIN, Species.ANNET),
        )
        val otherInput = EditText(this).apply {
            hint = Species.ANNET.label
            setText(speciesOther)
            visibility = if (species == Species.ANNET) View.VISIBLE else View.GONE
        }
        speciesRows.forEach { rowSpecies ->
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            rowSpecies.forEach { s ->
                row.addView(MaterialButton(this, null,
                    com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                    text = s.label
                    alpha = if (species == s) 1f else 0.55f
                    layoutParams = LinearLayout.LayoutParams(0,
                        Ui.dp(this@HuntLogActivity, 56), 1f).apply {
                        setMargins(Ui.dp(this@HuntLogActivity, 2), Ui.dp(this@HuntLogActivity, 2),
                            Ui.dp(this@HuntLogActivity, 2), Ui.dp(this@HuntLogActivity, 2))
                    }
                    setOnClickListener {
                        species = s
                        speciesOther = otherInput.text.toString().trim()
                        distanceM = distInput.text.toString().toIntOrNull()
                        page1()
                    }
                })
            }
            content.addView(row)
        }
        content.addView(otherInput)

        // Avstand: maks 4 sifre
        val distRow = Ui.row(this)
        distRow.addView(TextView(this).apply {
            text = "${getString(R.string.hunt_distance)}: "
            textSize = 16f
        })
        distRow.addView(distInput)
        distRow.addView(TextView(this).apply { text = " m"; textSize = 16f })
        content.addView(distRow)

        content.addView(Ui.hint(this, getString(R.string.hunt_edit_hint)))

        content.addView(MaterialButton(this).apply {
            text = getString(R.string.hunt_next)
            layoutParams = Ui.matchWrap(12, this@HuntLogActivity)
                .apply { height = Ui.dp(this@HuntLogActivity, 56) }
            setOnClickListener {
                speciesOther = otherInput.text.toString().trim()
                distanceM = distInput.text.toString().toIntOrNull()
                if (species == null) {
                    Toast.makeText(this@HuntLogActivity,
                        getString(R.string.hunt_step_species), Toast.LENGTH_SHORT).show()
                } else {
                    page2()
                }
            }
        })

        root.removeAllViews()
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
    }

    // ---------- Side 2 ----------

    private fun page2() {
        val content = Ui.col(this)

        // «Posisjon 1–6» i klokkeform rundt jaktikonet (vinkling-placeholder)
        val clockWrap = FrameLayout(this).apply {
            layoutParams = Ui.matchWrap(0, this@HuntLogActivity)
                .apply { height = Ui.dp(this@HuntLogActivity, 300) }
        }
        clockWrap.addView(ImageView(this).apply {
            setImageResource(R.drawable.ic_menu_moose)
            adjustViewBounds = true
        }, FrameLayout.LayoutParams(Ui.dp(this, 110), Ui.dp(this, 72), Gravity.CENTER))
        val radius = Ui.dp(this, 110).toFloat()
        (1..6).forEach { p ->
            val angle = Math.toRadians((p - 1) * 60.0 - 90.0)   // 1 øverst, med klokka
            clockWrap.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = getString(R.string.hunt_pos_label, p)
                textSize = 12f
                alpha = if (clockPos == p) 1f else 0.55f
                translationX = (radius * cos(angle)).toFloat()
                translationY = (radius * sin(angle)).toFloat()
                setOnClickListener { clockPos = p; page2() }
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.CENTER))
        }
        content.addView(clockWrap)

        // Utfall: «Dyret løp X m» / Ettersøk / Bomskudd
        val ranRow = Ui.row(this)
        ranRow.addView(TextView(this).apply {
            text = "${getString(R.string.hunt_ran)} "
            textSize = 16f
        })
        val ranInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            filters = arrayOf(InputFilter.LengthFilter(4))
            minWidth = Ui.dp(this@HuntLogActivity, 70)
        }
        ranRow.addView(ranInput)
        ranRow.addView(TextView(this).apply { text = " m   "; textSize = 16f })
        var ettersok = false
        var bom = false
        lateinit var ettersokBtn: MaterialButton
        lateinit var bomBtn: MaterialButton
        fun styleOutcome() {
            ettersokBtn.alpha = if (ettersok) 1f else 0.55f
            bomBtn.alpha = if (bom) 1f else 0.55f
        }
        ettersokBtn = MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.hunt_ettersok)
            setOnClickListener { ettersok = !ettersok; if (ettersok) bom = false
                styleOutcome() }
        }
        bomBtn = MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.hunt_bom)
            setOnClickListener { bom = !bom; if (bom) ettersok = false
                styleOutcome() }
        }
        styleOutcome()
        ranRow.addView(ettersokBtn)
        ranRow.addView(bomBtn)
        content.addView(ranRow)

        val btnRow = Ui.row(this)
        btnRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.cancel)
            setOnClickListener { finish() }
        })
        btnRow.addView(Space(this), LinearLayout.LayoutParams(0, 1, 1f))
        btnRow.addView(MaterialButton(this).apply {
            text = getString(R.string.hunt_register)
            setOnClickListener {
                save(ranInput.text.toString().toIntOrNull(), ettersok, bom)
            }
        })
        content.addView(btnRow, Ui.matchWrap(16, this))

        root.removeAllViews()
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
    }

    private fun save(ranM: Int?, ettersok: Boolean, bom: Boolean) {
        val outcome = when {
            bom -> Outcome.BOM
            ettersok -> Outcome.SKADE
            // «Dødelig» = dyret løp kortere enn ~100 m (spec §4/§8)
            ranM != null && ranM <= 100 -> Outcome.DOEDELIG
            ranM != null -> Outcome.SKADE
            else -> Outcome.SKADE
        }
        // Klokkeposisjon -> provisorisk vinkelkategori (spec §10.2)
        val angle = when (clockPos) {
            1 -> Angle.FRONT; 2, 6 -> Angle.SKRAA30
            3, 5 -> Angle.SIDE; 4 -> Angle.BAK
            else -> Angle.SIDE
        }
        store.addHunt(HuntRecord(
            id = Store.newId(),
            ts = System.currentTimeMillis(),
            species = species ?: Species.ANNET,
            distanceM = distanceM ?: 0,
            angle = angle,
            moving = false,
            outcome = outcome,
            lat = lat, lon = lon,
            weaponId = store.selectedWeapon()?.id,
            placeName = placeName,
            speciesOther = speciesOther,
            ranM = ranM,
            clockPos = clockPos,
        ))
        Toast.makeText(this, R.string.hunt_saved, Toast.LENGTH_SHORT).show()
        finish()
    }
}
