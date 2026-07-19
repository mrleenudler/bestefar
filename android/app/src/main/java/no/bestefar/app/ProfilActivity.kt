package no.bestefar.app

import android.os.Bundle
import android.text.InputType
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

/**
 * Profil (spec §6): skytterprofil m/skadeskytingsrate, våpenkartotek,
 * data og samtykke, sletting. Én skytter per installasjon antas.
 */
class ProfilActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var content: LinearLayout

    private val rates = listOf(
        0.02 to "1 av 50 (2 %)",
        0.05 to "1 av 20 (5 %)",
        0.10 to "1 av 10 (10 %)",
    )
    private val granularities = arrayOf(
        "Presis (kun lokalt)", "Kommune (~10 km)", "Fylke (~50 km)", "Ingen deling")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        setContentView(Ui.scroll(this, content))
        rebuild()
    }

    private fun rebuild() {
        content.removeAllViews()
        content.addView(Ui.title(this, getString(R.string.profile_title)))

        // ---------- Skytterprofil ----------
        content.addView(Ui.section(this, getString(R.string.profile_shooter)))
        val birth = EditText(this).apply {
            hint = getString(R.string.birth_year_hint)
            inputType = InputType.TYPE_CLASS_NUMBER
            setText(if (store.birthYear == 0) "" else store.birthYear.toString())
        }
        val lag = EditText(this).apply {
            hint = getString(R.string.skytterlag_hint)
            setText(store.skytterlag)
        }
        val jaktlag = EditText(this).apply {
            hint = getString(R.string.jaktlag_hint)
            setText(store.jaktlag)
        }
        content.addView(birth); content.addView(lag); content.addView(jaktlag)
        content.addView(MaterialButton(this).apply {
            text = getString(R.string.save)
            setOnClickListener {
                store.birthYear = birth.text.toString().toIntOrNull() ?: 0
                store.skytterlag = lag.text.toString().trim()
                store.jaktlag = jaktlag.text.toString().trim()
                Toast.makeText(this@ProfilActivity, R.string.save, Toast.LENGTH_SHORT).show()
            }
        })

        content.addView(Ui.body(this, getString(R.string.profile_rate)))
        content.addView(Ui.hint(this, getString(R.string.profile_rate_info)))
        val group = RadioGroup(this)
        rates.forEach { (rate, label) ->
            group.addView(RadioButton(this).apply {
                text = label
                isChecked = kotlin.math.abs(store.rateLimit - rate) < 0.001
                setOnClickListener {
                    // Endring virker umiddelbart (spec §6); hendelseslogg TODO
                    store.rateLimit = rate
                }
            })
        }
        content.addView(group)

        // ---------- Våpenkartotek ----------
        content.addView(Ui.section(this, getString(R.string.profile_weapons)))
        store.weapons().forEach { w ->
            val row = Ui.row(this)
            val desc = buildString {
                append(w.shownName)
                store.clickCmFor(w)?.let { append(" · %.2f cm/klikk".format(it)) }
                if (w.ammoSplit) append(" · ammosplitt")
                if (w.ammoName.isNotBlank()) append(" · ${w.ammoName}")
            }
            row.addView(Ui.body(this, desc).apply {
                layoutParams = LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            })
            row.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = "Endre"
                setOnClickListener {
                    Dialogs.weaponEdit(this@ProfilActivity, store, w, true) { rebuild() }
                }
            })
            content.addView(row)
        }
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.weapon_add)
            setOnClickListener {
                Dialogs.weaponEdit(this@ProfilActivity, store, null, true) { rebuild() }
            }
        })

        // ---------- Data og samtykke ----------
        content.addView(Ui.section(this, getString(R.string.profile_data)))
        consentRow(getString(R.string.consent_research_title),
            store.consentResearch, research = true)
        consentRow(getString(R.string.consent_hunt_title),
            store.consentHunt, research = false)

        val granRow = Ui.row(this)
        granRow.addView(Ui.body(this,
            "${getString(R.string.profile_granularity)}: ${store.shareGranularity}").apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        granRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "Endre"
            setOnClickListener {
                AlertDialog.Builder(this@ProfilActivity)
                    .setTitle(R.string.profile_granularity)
                    .setItems(granularities) { _, i ->
                        store.shareGranularity = granularities[i]; rebuild()
                    }.show()
            }
        })
        content.addView(granRow)

        content.addView(Ui.hint(this, getString(R.string.profile_backup_status)))
        content.addView(Ui.hint(this, getString(R.string.profile_login)))

        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.profile_move)
            setOnClickListener {
                Toast.makeText(this@ProfilActivity, R.string.profile_move_todo,
                    Toast.LENGTH_SHORT).show()
            }
        })
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.profile_delete)
            setOnClickListener {
                AlertDialog.Builder(this@ProfilActivity)
                    .setMessage(R.string.profile_delete_confirm)
                    .setPositiveButton(R.string.profile_delete) { _, _ ->
                        store.wipeAll(); finish()
                    }
                    .setNegativeButton(R.string.cancel, null)
                    .show()
            }
        })
    }

    private fun consentRow(label: String, state: String, research: Boolean) {
        val stateLabel = when (state) {
            "ja" -> getString(R.string.consent_yes)
            "senere" -> getString(R.string.consent_later)
            "aldri" -> getString(R.string.consent_never)
            else -> "ikke besvart"
        }
        val row = Ui.row(this)
        row.addView(Ui.body(this, "$label $stateLabel").apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        row.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "Endre"
            setOnClickListener {
                val options = arrayOf(getString(R.string.consent_yes),
                    getString(R.string.consent_later), getString(R.string.consent_never))
                AlertDialog.Builder(this@ProfilActivity)
                    .setTitle(label)
                    .setItems(options) { _, i ->
                        when (i) {
                            0 -> if (research)
                                Dialogs.researchConsentYes(this@ProfilActivity, store) { rebuild() }
                            else { store.consentHunt = "ja"; rebuild() }
                            1 -> { if (research) store.consentResearch = "senere"
                                   else store.consentHunt = "senere"; rebuild() }
                            2 -> { if (research) store.consentResearch = "aldri"
                                   else store.consentHunt = "aldri"; rebuild() }
                        }
                    }.show()
            }
        })
        content.addView(row)
    }
}
