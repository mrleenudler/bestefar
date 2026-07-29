package no.bestefar.app

import android.Manifest
import android.annotation.SuppressLint
import android.app.DatePickerDialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
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
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * Jaktlogg (musingsUI runde 4), to sider:
 *  1) «Del med forskning»-checkbox (av = forenklet visning) → vilt → posisjon
 *     (rød pin) → avstand → Neste. Kalender-datovelger, dagens dato forhåndsvalgt.
 *  2) tre silhuetter av valgt vilt for «Velg dyrets posisjon» → «Dyret løp ca
 *     X m» → Ettersøk/Bomskudd → Avbryt / Registrer skudd.
 */
class HuntLogActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var root: FrameLayout

    private var lat: Double? = null
    private var lon: Double? = null
    private var placeName: String = ""
    private var species: Species? = null
    private var speciesOther: String = ""
    private var distanceM: Int? = null
    private var posAsked = false
    private var shareResearch = false
    private var selectedDate: LocalDate = LocalDate.now()
    private var posSel: Int? = null           // 1=front, 2=side, 3=skrå

    private val dateFmt = DateTimeFormatter.ofPattern("d. MMMM yyyy", Locale("no"))

    private fun night() = (resources.configuration.uiMode and
        android.content.res.Configuration.UI_MODE_NIGHT_MASK) ==
        android.content.res.Configuration.UI_MODE_NIGHT_YES
    /** Silhuett-tint: tekstfarge i mørk visning, sort i lys (musingsUI runde 7). */
    private fun silTint() = if (night())
        Ui.themeColor(this, android.R.attr.textColorPrimary) else Color.BLACK

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        // «Del med forskning» reflekterer tidligere valg (musingsUI runde 4)
        shareResearch = store.consentHunt == "ja"
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
        Thread {
            try {
                @Suppress("DEPRECATION")
                val res = Geocoder(this, Locale.getDefault())
                    .getFromLocation(loc.latitude, loc.longitude, 1)
                val name = res?.firstOrNull()?.let {
                    it.subAdminArea ?: it.locality ?: it.adminArea
                } ?: ""
                runOnUiThread {
                    if (name.isNotBlank() && placeName.isBlank()) { placeName = name; page1() }
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
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun manualPlaceDialog() {
        val input = EditText(this).apply { hint = getString(R.string.hunt_position) }
        Ui.capitalize(input)
        AlertDialog.Builder(this)
            .setTitle(R.string.hunt_pos_manual)
            .setView(input)
            .setPositiveButton(R.string.ok) { _, _ ->
                placeName = input.text.toString().trim(); page1()
            }
            .show()
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>,
                                            results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == 3 && results.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            tryFetchLocation(); page1()
        }
    }

    // ---------- Side 1 ----------

    private fun speciesButtons(otherInput: EditText, content: LinearLayout,
                               distInput: EditText?) {
        listOf(
            listOf(Species.RAADYR, Species.HJORT, Species.ELG),
            listOf(Species.VILLREIN, Species.VILLSVIN, Species.ANNET),
        ).forEach { rowSpecies ->
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            rowSpecies.forEach { s ->
                row.addView(Ui.choiceButton(this, s.label, species == s) {
                    species = s
                    speciesOther = otherInput.text.toString().trim()
                    distInput?.let { distanceM = it.text.toString().toIntOrNull() }
                    // «Annet» -> markør i tekstboks + deaktiver forskningsdeling
                    if (s == Species.ANNET) shareResearch = false
                    page1()
                    if (s == Species.ANNET) {
                        otherInput.requestFocus()
                        otherInput.setSelection(otherInput.text.length)
                    }
                }.apply {
                    layoutParams = LinearLayout.LayoutParams(0,
                        Ui.dp(this@HuntLogActivity, 56), 1f).apply {
                        setMargins(Ui.dp(this@HuntLogActivity, 2), Ui.dp(this@HuntLogActivity, 2),
                            Ui.dp(this@HuntLogActivity, 2), Ui.dp(this@HuntLogActivity, 2))
                    }
                })
            }
            content.addView(row)
        }
        content.addView(otherInput)
    }

    private fun dateButton(): MaterialButton = MaterialButton(this, null,
        com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
        text = selectedDate.format(dateFmt)
        layoutParams = Ui.matchWrap(4, this@HuntLogActivity)
        setOnClickListener {
            DatePickerDialog(this@HuntLogActivity, { _, y, m, d ->
                selectedDate = LocalDate.of(y, m + 1, d)
                text = selectedDate.format(dateFmt)
            }, selectedDate.year, selectedDate.monthValue - 1,
                selectedDate.dayOfMonth).show()
        }
    }

    private fun page1() {
        val content = Ui.col(this)

        content.addView(android.widget.CheckBox(this).apply {
            text = getString(R.string.hunt_share_research)
            isChecked = shareResearch
            isEnabled = species != Species.ANNET
            setOnCheckedChangeListener { _, on -> shareResearch = on; page1() }
        })

        val otherInput = EditText(this).apply {
            hint = getString(R.string.hunt_other_hint)
            visibility = if (species == Species.ANNET) View.VISIBLE else View.GONE
        }
        Ui.capitalize(otherInput)
        Ui.boxed(otherInput, this)
        otherInput.setText(speciesOther)

        content.addView(dateButton())
        speciesButtons(otherInput, content, null)

        if (!shareResearch) {
            // Forenklet visning: dato + vilt + «Felling var vellykket» + Lagre/Avbryt.
            // Felling-checkboxen hører hjemme også her (musingsUI runde 7).
            val fellingBox = android.widget.CheckBox(this).apply {
                text = getString(R.string.felling_success); isChecked = true
            }
            content.addView(fellingBox)

            val btnRow = Ui.row(this)
            btnRow.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = getString(R.string.cancel)
                setOnClickListener { finish() }
            })
            btnRow.addView(Space(this), LinearLayout.LayoutParams(0, 1, 1f))
            btnRow.addView(MaterialButton(this).apply {
                text = getString(R.string.save)
                setOnClickListener {
                    speciesOther = otherInput.text.toString().trim()
                    when {
                        species == null ->
                            Ui.toast(this@HuntLogActivity, R.string.hunt_missing_species)
                        species == Species.ANNET && speciesOther.length < 2 ->
                            Ui.toast(this@HuntLogActivity, R.string.hunt_missing_other)
                        // Uavkrysset «felling vellykket» -> bekreft (runde 6/7)
                        !fellingBox.isChecked -> AlertDialog.Builder(this@HuntLogActivity)
                            .setMessage(R.string.felling_ask)
                            .setPositiveButton(R.string.yes) { _, _ -> saveSimple(true) }
                            .setNegativeButton(R.string.no) { _, _ -> saveSimple(false) }
                            .show()
                        else -> saveSimple(true)
                    }
                }
            })
            content.addView(btnRow, Ui.matchWrap(16, this))
        } else {
            val distInput = EditText(this).apply {
                inputType = InputType.TYPE_CLASS_NUMBER
                filters = arrayOf(InputFilter.LengthFilter(4), Ui.noLeadingZero())
                minWidth = Ui.dp(this@HuntLogActivity, 90)
                setText(distanceM?.toString() ?: "")
            }
            Ui.boxed(distInput, this)

            // Posisjon under viltknappene, med rød pin (musingsUI runde 3/4)
            val posText = when {
                placeName.isNotBlank() && lat != null -> "$placeName (%.4f, %.4f)".format(lat, lon)
                placeName.isNotBlank() -> placeName
                lat != null -> "%.4f, %.4f".format(lat, lon)
                else -> "—"
            }
            val posRow = Ui.row(this)
            posRow.addView(ImageView(this).apply {
                setImageResource(R.drawable.ic_pin)
                layoutParams = LinearLayout.LayoutParams(
                    Ui.dp(this@HuntLogActivity, 24), Ui.dp(this@HuntLogActivity, 24))
                contentDescription = getString(R.string.hunt_position)
            })
            posRow.addView(TextView(this).apply { text = " $posText"; textSize = 16f })
            content.addView(posRow, Ui.matchWrap(8, this))
            if (lat == null && placeName.isBlank()) {
                if (!hasLocationPermission()) askForPosition()
                content.addView(MaterialButton(this, null,
                    com.google.android.material.R.attr.borderlessButtonStyle).apply {
                    text = getString(R.string.hunt_pos_manual)
                    setOnClickListener { manualPlaceDialog() }
                })
            }

            val distRow = Ui.row(this)
            distRow.addView(TextView(this).apply {
                text = "${getString(R.string.hunt_hold)}: "; textSize = 17f
            })
            distRow.addView(distInput)
            distRow.addView(TextView(this).apply { text = " m"; textSize = 17f })
            content.addView(distRow, Ui.matchWrap(8, this))

            content.addView(MaterialButton(this).apply {
                text = getString(R.string.hunt_next)
                layoutParams = Ui.matchWrap(12, this@HuntLogActivity)
                    .apply { height = Ui.dp(this@HuntLogActivity, 56) }
                setOnClickListener {
                    speciesOther = otherInput.text.toString().trim()
                    distanceM = distInput.text.toString().toIntOrNull()
                    when {
                        species == null || distanceM == null ->
                            Ui.toast(this@HuntLogActivity, R.string.hunt_missing_species_distance)
                        species == Species.ANNET && speciesOther.length < 2 ->
                            Ui.toast(this@HuntLogActivity, R.string.hunt_missing_other)
                        else -> page2()
                    }
                }
            })
            // «Informasjonen kan redigeres senere» UNDER Neste (musingsUI runde 4)
            content.addView(TextView(this).apply {
                text = getString(R.string.hunt_edit_hint)
                textSize = 15f; alpha = 0.75f
                setPadding(0, Ui.dp(this@HuntLogActivity, 6), 0, 0)
            })
        }

        root.removeAllViews()
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
    }

    private fun tsForDate(): Long =
        selectedDate.atTime(12, 0).atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()

    /**
     * Forenklet logging: dato + vilt, lagres som felling (dødelig). Posisjon
     * logges også her (musingsUI runde 7 — tidligere kun ved forskningsdeling).
     */
    private fun saveSimple(fellingSuccess: Boolean) {
        store.addHunt(HuntRecord(
            id = Store.newId(), ts = tsForDate(),
            species = species ?: Species.ANNET, distanceM = 0,
            angle = Angle.SIDE, moving = false, outcome = Outcome.DOEDELIG,
            lat = lat, lon = lon, placeName = placeName,
            weaponId = store.selectedWeapon()?.id, speciesOther = speciesOther,
            created = System.currentTimeMillis(), fellingSuccess = fellingSuccess,
        ))
        Ui.toast(this, R.string.hunt_saved)
        finish()
    }

    // ---------- Side 2 ----------

    private fun page2() {
        val content = Ui.col(this)

        // Tre silhuetter av valgt vilt (hjort brukes på alle inntil videre)
        val silRow = Ui.row(this).apply { gravity = Gravity.CENTER }
        val sils = listOf(
            1 to R.drawable.ic_hjort_front,
            2 to R.drawable.ic_hjort_side,
            3 to R.drawable.ic_hjort_skraa)
        sils.forEach { (pos, res) ->
            val iv = ImageView(this).apply {
                setImageResource(res)
                scaleType = ImageView.ScaleType.FIT_CENTER
                androidx.core.widget.ImageViewCompat.setImageTintList(this,
                    android.content.res.ColorStateList.valueOf(silTint()))
                val p = Ui.dp(this@HuntLogActivity, 6)
                setPadding(p, p, p, p)
                background = GradientDrawable().apply {
                    cornerRadius = Ui.dp(this@HuntLogActivity, 8).toFloat()
                    setStroke(Ui.dp(this@HuntLogActivity, 2),
                        if (posSel == pos)
                            Ui.themeColor(this@HuntLogActivity,
                                com.google.android.material.R.attr.colorPrimary)
                        else Color.TRANSPARENT)
                }
                layoutParams = LinearLayout.LayoutParams(0,
                    Ui.dp(this@HuntLogActivity, 130), 1f)
                setOnClickListener { posSel = pos; page2() }
            }
            silRow.addView(iv)
        }
        content.addView(silRow)

        val ranRow = Ui.row(this)
        ranRow.addView(TextView(this).apply {
            text = "${getString(R.string.hunt_ran)} "; textSize = 17f
        })
        val ranInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            // Godta 0 m, men ikke sifre etter en enslig 0 (musingsUI runde 6)
            filters = arrayOf(InputFilter.LengthFilter(4), Ui.singleZero())
            minWidth = Ui.dp(this@HuntLogActivity, 80)
        }
        Ui.boxed(ranInput, this)
        ranRow.addView(ranInput)
        ranRow.addView(TextView(this).apply { text = " m"; textSize = 17f })
        content.addView(ranRow, Ui.matchWrap(8, this))

        // «Felling var vellykket» (musingsUI runde 6)
        val fellingBox = android.widget.CheckBox(this).apply {
            text = getString(R.string.felling_success); isChecked = true
        }
        content.addView(fellingBox)

        var ettersok = false
        var bom = false
        val outcomeRow = Ui.row(this)
        val notFoundBox = android.widget.CheckBox(this).apply {
            text = getString(R.string.hunt_not_found); visibility = View.GONE
        }
        fun renderOutcome() {
            outcomeRow.removeAllViews()
            outcomeRow.addView(Ui.choiceButton(this, getString(R.string.hunt_ettersok),
                ettersok) {
                ettersok = !ettersok
                if (ettersok) bom = false
                notFoundBox.visibility = if (ettersok) View.VISIBLE else View.GONE
                renderOutcome()
            }.apply { layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f) })
            outcomeRow.addView(Space(this), LinearLayout.LayoutParams(Ui.dp(this, 8), 1))
            outcomeRow.addView(Ui.choiceButton(this, getString(R.string.hunt_bom), bom) {
                bom = !bom
                if (bom) { ettersok = false; notFoundBox.visibility = View.GONE }
                renderOutcome()
            }.apply { layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f) })
        }
        renderOutcome()
        content.addView(outcomeRow, Ui.matchWrap(8, this))
        content.addView(notFoundBox)

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
                val ran = ranInput.text.toString().toIntOrNull()
                // Bom eller «ikke funnet» krever ikke tall (musingsUI runde 5)
                val needRan = !bom && !notFoundBox.isChecked
                when {
                    needRan && ran == null ->
                        Ui.toast(this@HuntLogActivity, R.string.hunt_missing_ran)
                    // Uavkrysset «felling vellykket» -> bekreft (musingsUI runde 6)
                    !fellingBox.isChecked -> AlertDialog.Builder(this@HuntLogActivity)
                        .setMessage(R.string.felling_ask)
                        .setPositiveButton(R.string.yes) { _, _ ->
                            save(ran, ettersok, bom, notFoundBox.isChecked, true) }
                        .setNegativeButton(R.string.no) { _, _ ->
                            save(ran, ettersok, bom, notFoundBox.isChecked, false) }
                        .show()
                    else -> save(ran, ettersok, bom, notFoundBox.isChecked, true)
                }
            }
        })
        content.addView(btnRow, Ui.matchWrap(16, this))

        root.removeAllViews()
        root.addView(Ui.scroll(this, content), ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT)
    }

    private fun save(ranM: Int?, ettersok: Boolean, bom: Boolean, notFound: Boolean,
                     fellingSuccess: Boolean) {
        val outcome = when {
            bom -> Outcome.BOM
            ettersok -> Outcome.SKADE
            ranM != null && ranM <= 100 -> Outcome.DOEDELIG
            ranM != null -> Outcome.SKADE
            else -> Outcome.SKADE
        }
        val angle = when (posSel) {
            1 -> Angle.FRONT; 3 -> Angle.SKRAA30; else -> Angle.SIDE
        }
        val now = System.currentTimeMillis()
        store.addHunt(HuntRecord(
            id = Store.newId(), ts = tsForDate(),
            species = species ?: Species.ANNET, distanceM = distanceM ?: 0,
            angle = angle, moving = false, outcome = outcome,
            followUp = if (ettersok && notFound) FollowUp.IKKE_GJENFUNNET else null,
            lat = lat, lon = lon, weaponId = store.selectedWeapon()?.id,
            placeName = placeName, speciesOther = speciesOther,
            ranM = ranM, clockPos = posSel,
            created = now, fellingSuccess = fellingSuccess,
        ))
        Ui.toast(this, R.string.hunt_saved)
        finish()
    }
}
