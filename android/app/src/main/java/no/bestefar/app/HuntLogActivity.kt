package no.bestefar.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Bundle
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.slider.Slider

/**
 * Hurtiglogg — tre steg, hanskevennlig (spec §4): art → hold+vinkling →
 * utfall. Mål: under et halvt minutt, helt uten dekning. Systemhentede
 * metadata (tid, ev. posisjon) legges på automatisk.
 */
class HuntLogActivity : AppCompatActivity() {

    private lateinit var content: LinearLayout
    private lateinit var store: Store

    private var species = Species.ELG
    private var holdM = 80
    private var angle = Angle.SIDE
    private var moving = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        setContentView(Ui.scroll(this, content))
        step1()
    }

    private fun bigButton(label: String, onClick: () -> Unit) = MaterialButton(this).apply {
        text = label
        textSize = 18f
        layoutParams = Ui.matchWrap(8, this@HuntLogActivity)
            .apply { height = Ui.dp(this@HuntLogActivity, 64) }
        setOnClickListener { onClick() }
    }

    /** Steg 1: art (ikonene kommer; store tekstknapper er hanskevennlige). */
    private fun step1() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.hunt_step_species)))
        val dagens = store.huntSpecies
        val ordered = Species.entries.sortedByDescending { it.name in dagens }
        ordered.forEach { s ->
            content.addView(bigButton(s.label) { species = s; step2() })
        }
    }

    /** Steg 2: hold + vinkling (dyr-i-sirkel-grafikk kommer, spec §4). */
    private fun step2() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.hunt_step_hold)))

        val holdText = TextView(this).apply { textSize = 26f; text = "$holdM m" }
        content.addView(holdText)
        content.addView(Slider(this).apply {
            valueFrom = 10f; valueTo = 400f; stepSize = 5f
            value = holdM.toFloat()
            addOnChangeListener { _, v, _ ->
                holdM = v.toInt(); holdText.text = "$holdM m"
            }
        })

        val angleRow = Ui.row(this)
        angleRow.addView(TextView(this).apply {
            text = "Vinkling"; textSize = 16f
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        })
        angleRow.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_info)
            background = null
            contentDescription = "Forklaring av vinkling"
            setOnClickListener {
                AlertDialog.Builder(this@HuntLogActivity)
                    .setMessage(R.string.hunt_angle_info)
                    .setPositiveButton(R.string.ok, null).show()
            }
        })
        content.addView(angleRow)

        val chips = ChipGroup(this).apply { isSingleSelection = true }
        Angle.entries.forEach { a ->
            chips.addView(Chip(this).apply {
                text = a.label; isCheckable = true; isChecked = a == angle
                setOnClickListener { angle = a }
            })
        }
        content.addView(chips)

        content.addView(Chip(this).apply {
            text = getString(R.string.hunt_moving)
            isCheckable = true; isChecked = moving
            setOnClickListener { moving = isChecked }
        })

        content.addView(bigButton(getString(R.string.hunt_next)) { step3() })
    }

    /** Steg 3: utfall — dødelig (operasjonalisert), skade, bom (spec §4). */
    private fun step3() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.hunt_step_outcome)))
        content.addView(bigButton(
            "${Outcome.DOEDELIG.label} — ${getString(R.string.hunt_outcome_lethal_sub)}") {
            save(Outcome.DOEDELIG)
        })
        content.addView(bigButton(Outcome.SKADE.label) { save(Outcome.SKADE) })
        content.addView(bigButton(Outcome.BOM.label) { save(Outcome.BOM) })
    }

    @SuppressLint("MissingPermission")
    private fun save(outcome: Outcome) {
        var lat: Double? = null
        var lon: Double? = null
        // Presis posisjon lagres KUN lokalt når påslått (spec §4)
        if (store.locationLogging && ContextCompat.checkSelfPermission(this,
                Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            val lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
            val loc = try {
                lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
                    ?: lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            } catch (_: Exception) { null }
            lat = loc?.latitude; lon = loc?.longitude
        }
        store.addHunt(HuntRecord(
            id = Store.newId(),
            ts = System.currentTimeMillis(),
            species = species,
            distanceM = holdM,
            angle = angle,
            moving = moving,
            outcome = outcome,
            lat = lat, lon = lon,
            weaponId = store.selectedWeapon()?.id,
        ))
        Toast.makeText(this, R.string.hunt_saved, Toast.LENGTH_SHORT).show()
        finish()
    }
}
