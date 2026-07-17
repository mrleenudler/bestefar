package no.bestefar.app

import android.app.Activity
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import java.time.LocalDate
import kotlin.math.floor

object Dialogs {

    private fun dp(a: Activity, v: Int) = (v * a.resources.displayMetrics.density).toInt()

    /**
     * Stillingsprompt etter skivescan (spec §2): fire hovedstillinger med
     * modifikator-chips (uten/anlegg/reim, siste huskes per stilling) +
     * Benk som egen inngang.
     */
    fun positionSheet(a: Activity, store: Store, onChosen: (Position, PosModifier) -> Unit) {
        val sheet = BottomSheetDialog(a)
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 20), dp(a, 20), dp(a, 20), dp(a, 24))
        }
        root.addView(TextView(a).apply {
            text = a.getString(R.string.position_prompt_title)
            textSize = 20f
        })

        val chips = ChipGroup(a).apply { isSingleSelection = true }
        var selectedPos = store.currentPosition.takeIf { it != Position.BENK } ?: Position.LIGGENDE
        val modChips = PosModifier.entries.map { m ->
            Chip(a).apply {
                text = m.label; isCheckable = true; tag = m
            }
        }
        modChips.forEach { chips.addView(it) }
        fun syncMods() {
            val remembered = store.lastModifier(selectedPos)
            modChips.forEach { it.isChecked = it.tag == remembered }
        }

        val posButtons = Position.hoved.map { p ->
            MaterialButton(a).apply {
                text = p.label
                tag = p
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, dp(a, 56)
                ).apply { topMargin = dp(a, 8) }
            }
        }
        fun styleButtons() {
            posButtons.forEach { b -> b.alpha = if (b.tag == selectedPos) 1f else 0.55f }
        }
        posButtons.forEach { b ->
            b.setOnClickListener {
                selectedPos = b.tag as Position
                styleButtons(); syncMods()
            }
            root.addView(b)
        }

        root.addView(TextView(a).apply {
            text = a.getString(R.string.position_modifier)
            setPadding(0, dp(a, 12), 0, 0)
        })
        root.addView(chips)

        val row = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.END
            setPadding(0, dp(a, 12), 0, 0)
        }
        row.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = a.getString(R.string.position_benk)
            setOnClickListener {
                sheet.dismiss()
                onChosen(Position.BENK, PosModifier.ANLEGG)
            }
        })
        row.addView(MaterialButton(a).apply {
            text = a.getString(R.string.ok)
            setOnClickListener {
                val mod = modChips.firstOrNull { it.isChecked }?.tag as? PosModifier
                    ?: PosModifier.UTEN
                store.setLastModifier(selectedPos, mod)
                store.currentPosition = selectedPos
                store.currentModifier = mod
                sheet.dismiss()
                onChosen(selectedPos, mod)
            }
        })
        root.addView(row)

        styleButtons(); syncMods()
        sheet.setContentView(root)
        sheet.setCancelable(false)
        sheet.show()
    }

    /** Dagsbekreftelse av våpen (+ ammo ved split) — én gang per dag (spec §2). */
    fun weaponDayConfirm(a: Activity, store: Store, onDone: () -> Unit) {
        val ws = store.weapons()
        if (ws.size <= 1 || !store.weaponNeedsDayConfirm()) { onDone(); return }
        val names = ws.map { it.name }.toTypedArray()
        var idx = ws.indexOfFirst { it.id == store.selectedWeaponId }.coerceAtLeast(0)
        AlertDialog.Builder(a)
            .setTitle(R.string.weapon_confirm_title)
            .setSingleChoiceItems(names, idx) { _, i -> idx = i }
            .setPositiveButton(R.string.ok) { _, _ ->
                val w = ws[idx]
                store.selectedWeaponId = w.id
                store.weaponConfirmedDate = LocalDate.now().toString()
                if (w.ammoSplit) confirmAmmo(a, store, w, onDone) else onDone()
            }
            .setCancelable(false)
            .show()
    }

    private fun confirmAmmo(a: Activity, store: Store, w: Weapon, onDone: () -> Unit) {
        val input = EditText(a).apply {
            setText(w.ammoName)
            hint = a.getString(R.string.ammo_hint)
        }
        AlertDialog.Builder(a)
            .setTitle(a.getString(R.string.ammo_confirm_title, w.name))
            .setView(input)
            .setPositiveButton(R.string.ok) { _, _ ->
                w.ammoName = input.text.toString().trim()
                store.updateWeapon(w)
                onDone()
            }
            .setCancelable(false)
            .show()
    }

    /**
     * Forskningssamtykke: Ja / Ikke nå / Aldri (spec §1). Tilbys etter fem
     * serier, deretter hver tiende ved «Ikke nå». 18-årsgrense (spec §7).
     */
    fun maybeResearchConsent(a: Activity, store: Store) {
        val n = store.allSeries().size
        val due = when (store.consentResearch) {
            "" -> n >= 5
            "senere" -> n - store.consentLastPromptCount >= 10
            else -> false
        }
        if (!due) return
        store.consentLastPromptCount = n
        AlertDialog.Builder(a)
            .setTitle(R.string.consent_research_title)
            .setMessage(R.string.consent_research_body)
            .setPositiveButton(R.string.consent_yes) { _, _ -> researchConsentYes(a, store) {} }
            .setNegativeButton(R.string.consent_later) { _, _ -> store.consentResearch = "senere" }
            .setNeutralButton(R.string.consent_never) { _, _ -> store.consentResearch = "aldri" }
            .show()
    }

    /** «Ja» til forskningssamtykke med 18-årsgate (spec §7). */
    fun researchConsentYes(a: Activity, store: Store, onDone: () -> Unit) {
        val year = LocalDate.now().year
        if (store.birthYear in 1..(year - 18)) {
            store.consentResearch = "ja"; onDone(); return
        }
        if (store.birthYear != 0) {
            AlertDialog.Builder(a).setMessage(R.string.consent_age_denied)
                .setPositiveButton(R.string.ok) { _, _ -> onDone() }.show()
            return
        }
        val input = EditText(a).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            hint = a.getString(R.string.birth_year_hint)
        }
        AlertDialog.Builder(a)
            .setTitle(R.string.consent_age_title)
            .setView(input)
            .setPositiveButton(R.string.ok) { _, _ ->
                val by = input.text.toString().toIntOrNull() ?: 0
                store.birthYear = by
                if (by in 1..(year - 18)) store.consentResearch = "ja"
                else AlertDialog.Builder(a).setMessage(R.string.consent_age_denied)
                    .setPositiveButton(R.string.ok, null).show()
                onDone()
            }
            .show()
    }

    /** Jaktsamtykke ved første bruk av jaktloggen (spec §7). */
    fun maybeHuntConsent(a: Activity, store: Store, onDone: () -> Unit) {
        if (store.consentHunt != "") { onDone(); return }
        AlertDialog.Builder(a)
            .setTitle(R.string.consent_hunt_title)
            .setMessage(R.string.consent_hunt_body)
            .setPositiveButton(R.string.consent_yes) { _, _ -> store.consentHunt = "ja"; onDone() }
            .setNegativeButton(R.string.consent_later) { _, _ -> store.consentHunt = "senere"; onDone() }
            .setNeutralButton(R.string.consent_never) { _, _ -> store.consentHunt = "aldri"; onDone() }
            .setCancelable(false)
            .show()
    }

    /** Legg til / endre våpen i kartoteket (spec §6). */
    fun weaponEdit(a: Activity, store: Store, existing: Weapon?, onDone: () -> Unit) {
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        val name = EditText(a).apply {
            hint = "Navn (f.eks. Sauer 100 6,5×55)"
            setText(existing?.name ?: "")
        }
        val click = EditText(a).apply {
            hint = a.getString(R.string.profile_click_hint)
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(existing?.clickValueCm?.toString() ?: "")
        }
        val ammo = EditText(a).apply {
            hint = a.getString(R.string.ammo_hint)
            setText(existing?.ammoName ?: "")
        }
        val split = android.widget.CheckBox(a).apply {
            text = a.getString(R.string.profile_ammo_split)
            isChecked = existing?.ammoSplit ?: false
        }
        root.addView(name); root.addView(click); root.addView(ammo); root.addView(split)
        AlertDialog.Builder(a)
            .setTitle(if (existing == null) R.string.weapon_add else R.string.profile_weapons)
            .setView(root)
            .setPositiveButton(R.string.save) { _, _ ->
                val n = name.text.toString().trim()
                if (n.isEmpty()) return@setPositiveButton
                val c = click.text.toString().replace(',', '.').toDoubleOrNull()
                if (existing == null) {
                    val w = Weapon(Store.newId(), n, c, split.isChecked,
                        ammo.text.toString().trim())
                    store.addWeapon(w)
                    if (store.weapons().size == 1) store.selectedWeaponId = w.id
                } else {
                    existing.name = n
                    existing.clickValueCm = c
                    existing.ammoSplit = split.isChecked
                    existing.ammoName = ammo.text.toString().trim()
                    store.updateWeapon(existing)
                }
                onDone()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    /**
     * Korrigering av skuddmerker (spec §2): poeng justeres i 0,1-steg per
     * skudd («som en timer-innstilling»). Ved lagring tilbys innsending til
     * feilanalysekanalen (mikrosamtykke per innsending).
     */
    fun correctionDialog(a: Activity, record: SeriesRecord, store: Store, onSaved: () -> Unit) {
        val values = record.shots.map { it.decimal }.toMutableList()
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        val labels = mutableListOf<TextView>()
        values.forEachIndexed { i, _ ->
            val row = LinearLayout(a).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            val label = TextView(a).apply {
                textSize = 20f
                layoutParams = LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            labels.add(label)
            fun render() { label.text = "Skudd ${i + 1}:  %.1f".format(values[i]) }
            render()
            row.addView(label)
            row.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = "−"; textSize = 22f
                setOnClickListener {
                    values[i] = (values[i] - 0.1).coerceAtLeast(0.0); render()
                }
            })
            row.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = "+"; textSize = 22f
                setOnClickListener {
                    values[i] = (values[i] + 0.1).coerceAtMost(10.9); render()
                }
            })
            root.addView(row)
        }
        AlertDialog.Builder(a)
            .setTitle(R.string.correction_title)
            .setView(root)
            .setPositiveButton(R.string.save) { _, _ ->
                record.shots = record.shots.mapIndexed { i, s ->
                    s.copy(decimal = values[i], integer = floor(values[i]).toInt().coerceAtMost(10))
                }
                record.corrected = true
                store.updateSeries(record)
                AlertDialog.Builder(a)
                    .setTitle(R.string.fail_channel_title)
                    .setMessage(R.string.fail_channel_body)
                    .setPositiveButton(R.string.send) { _, _ ->
                        record.sendToFailChannel = true
                        store.updateSeries(record)
                        onSaved()
                    }
                    .setNegativeButton(R.string.no_thanks) { _, _ -> onSaved() }
                    .show()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }
}
