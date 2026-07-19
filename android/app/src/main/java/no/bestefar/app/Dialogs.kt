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
    fun maybeResearchConsent(a: Activity, store: Store, onDone: () -> Unit = {}) {
        val n = store.allSeries().size
        val due = when (store.consentResearch) {
            "" -> n >= 5
            "senere" -> n - store.consentLastPromptCount >= 10
            else -> false
        }
        if (!due) { onDone(); return }
        store.consentLastPromptCount = n
        AlertDialog.Builder(a)
            .setTitle(R.string.consent_research_title)
            .setMessage(R.string.consent_research_body)
            .setPositiveButton(R.string.consent_yes) { _, _ -> researchConsentYes(a, store) {} }
            .setNegativeButton(R.string.consent_later) { _, _ -> store.consentResearch = "senere" }
            .setNeutralButton(R.string.consent_never) { _, _ -> store.consentResearch = "aldri" }
            .setOnDismissListener { onDone() }
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

    /**
     * Legg til / endre våpen (musingsUI): visningsnavn øverst, deretter
     * ikonvalg, så våpennavn med eksempel. `full` legger til optikkprofil-
     * kobling, ammo og ammosplitt (Meny → Rediger våpen).
     */
    fun weaponEdit(a: Activity, store: Store, existing: Weapon?, full: Boolean,
                   onDone: () -> Unit) {
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        val display = EditText(a).apply {
            hint = a.getString(R.string.weapon_display_hint)
            setText(existing?.displayName ?: "")
        }
        root.addView(display)

        var iconKey = existing?.icon ?: "rifle"
        val iconRow = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(a, 8), 0, dp(a, 8))
        }
        val iconViews = WeaponIcons.options.map { (key, res) ->
            android.widget.ImageButton(a).apply {
                setImageResource(res)
                background = null
                tag = key
                adjustViewBounds = true
                layoutParams = LinearLayout.LayoutParams(dp(a, 64), dp(a, 44))
                contentDescription = key
            }
        }
        fun styleIcons() = iconViews.forEach { it.alpha = if (it.tag == iconKey) 1f else 0.35f }
        iconViews.forEach { v ->
            v.setOnClickListener { iconKey = v.tag as String; styleIcons() }
            iconRow.addView(v)
        }
        styleIcons()
        root.addView(iconRow)

        val name = EditText(a).apply {
            hint = a.getString(R.string.weapon_name_hint)
            setText(existing?.name ?: "")
        }
        root.addView(name)

        var opticId = existing?.opticId
        var ammo: EditText? = null
        var split: android.widget.CheckBox? = null
        if (full) {
            val opticLabel = TextView(a).apply { setPadding(0, dp(a, 8), 0, 0) }
            fun renderOptic() {
                val o = store.optics().firstOrNull { it.id == opticId }
                opticLabel.text = a.getString(R.string.weapon_optic,
                    o?.let { "${it.displayName.ifBlank { it.brandModel }} (${it.reprLabel})" }
                        ?: "—")
            }
            renderOptic()
            root.addView(opticLabel)
            root.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = a.getString(R.string.weapon_optic_choose)
                setOnClickListener {
                    opticChooser(a, store) { chosen -> opticId = chosen; renderOptic() }
                }
            })
            ammo = EditText(a).apply {
                hint = a.getString(R.string.ammo_hint)
                setText(existing?.ammoName ?: "")
            }
            split = android.widget.CheckBox(a).apply {
                text = a.getString(R.string.profile_ammo_split)
                isChecked = existing?.ammoSplit ?: false
            }
            root.addView(ammo); root.addView(split)
        }

        AlertDialog.Builder(a)
            .setTitle(if (existing == null) R.string.weapon_add else R.string.weapon_edit)
            .setView(android.widget.ScrollView(a).apply { addView(root) })
            .setPositiveButton(R.string.save) { _, _ ->
                val n = name.text.toString().trim()
                val d = display.text.toString().trim()
                if (n.isEmpty() && d.isEmpty()) return@setPositiveButton
                if (existing == null) {
                    val w = Weapon(Store.newId(), n, null,
                        split?.isChecked ?: false, ammo?.text?.toString()?.trim() ?: "",
                        displayName = d, icon = iconKey, opticId = opticId)
                    store.addWeapon(w)
                    if (store.weapons().size == 1) store.selectedWeaponId = w.id
                } else {
                    existing.name = n
                    existing.displayName = d
                    existing.icon = iconKey
                    existing.opticId = opticId
                    ammo?.let { existing.ammoName = it.text.toString().trim() }
                    split?.let { existing.ammoSplit = it.isChecked }
                    store.updateWeapon(existing)
                }
                onDone()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    /** Velg blant optikkprofiler, eller opprett/rediger (musingsUI). */
    fun opticChooser(a: Activity, store: Store, onChosen: (String?) -> Unit) {
        val optics = store.optics()
        val items = optics.map {
            "${it.displayName.ifBlank { it.brandModel }} (${it.reprLabel})"
        } + a.getString(R.string.optic_new) + a.getString(R.string.optic_none)
        AlertDialog.Builder(a)
            .setTitle(R.string.optic_choose_title)
            .setItems(items.toTypedArray()) { _, i ->
                when {
                    i < optics.size -> onChosen(optics[i].id)
                    i == optics.size -> opticEdit(a, store, null) { onChosen(it.id) }
                    else -> onChosen(null)
                }
            }
            .show()
    }

    /**
     * Optikkprofil-editor (musingsUI-spec): visningsnavn + merke/modell,
     * radio for aktiv representasjon (MOA/MRAD/cm@100m) med ekspanderende
     * verdiliste; cm read-only avledet under MOA/MRAD; SMOA under avansert.
     */
    fun opticEdit(a: Activity, store: Store, existing: OpticProfile?,
                  onDone: (OpticProfile) -> Unit) {
        val o = existing ?: OpticProfile(Store.newId(), "", "", "MOA",
            0.25, 0.1, 1.0, false)
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        val display = EditText(a).apply {
            hint = a.getString(R.string.optic_display_hint)
            setText(o.displayName)
        }
        val brand = EditText(a).apply {
            hint = a.getString(R.string.optic_brand_hint)
            setText(o.brandModel)
        }
        root.addView(display); root.addView(brand)
        root.addView(TextView(a).apply {
            text = a.getString(R.string.optic_click_title)
            setPadding(0, dp(a, 12), 0, 0)
        })

        val derivedCm = TextView(a).apply { alpha = 0.7f }
        val cmInput = EditText(a).apply {
            hint = a.getString(R.string.profile_click_hint)
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setText(if (o.repr == "CM") o.cmValue.toString() else "")
        }
        val smoaBox = android.widget.CheckBox(a).apply {
            text = a.getString(R.string.optic_smoa)
            isChecked = o.smoa
        }

        val radios = listOf("MOA" to "MOA", "MRAD" to "MRAD", "CM" to "cm/100m")
            .map { (key, label) ->
                android.widget.RadioButton(a).apply { text = label; tag = key }
            }
        val moaButtons = OpticProfile.MOA_STEPS.map { v ->
            MaterialButton(a, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = "${OpticProfile.moaLabel(v)} MOA"; tag = v
            }
        }
        val mradButtons = OpticProfile.MRAD_STEPS.map { v ->
            MaterialButton(a, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = "%.2f mrad".format(v).replace('.', ','); tag = v
            }
        }
        val moaList = LinearLayout(a).apply { orientation = LinearLayout.VERTICAL }
        moaButtons.forEach { moaList.addView(it, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)) }
        val mradList = LinearLayout(a).apply { orientation = LinearLayout.VERTICAL }
        mradButtons.forEach { mradList.addView(it, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)) }

        fun sync() {
            radios.forEach { it.isChecked = it.tag == o.repr }
            moaList.visibility = if (o.repr == "MOA") android.view.View.VISIBLE
                else android.view.View.GONE
            mradList.visibility = if (o.repr == "MRAD") android.view.View.VISIBLE
                else android.view.View.GONE
            cmInput.visibility = if (o.repr == "CM") android.view.View.VISIBLE
                else android.view.View.GONE
            derivedCm.visibility = if (o.repr == "CM") android.view.View.GONE
                else android.view.View.VISIBLE
            moaButtons.forEach { it.alpha = if (it.tag == o.moaValue) 1f else 0.5f }
            mradButtons.forEach { it.alpha = if (it.tag == o.mradValue) 1f else 0.5f }
            o.smoa = smoaBox.isChecked
            derivedCm.text = a.getString(R.string.optic_derived_cm, o.clickCmPer100)
        }
        radios.forEachIndexed { i, rb ->
            rb.setOnClickListener { o.repr = rb.tag as String; sync() }
            root.addView(rb)
            when (i) {
                0 -> root.addView(moaList)
                1 -> root.addView(mradList)
                2 -> root.addView(cmInput)
            }
        }
        moaButtons.forEach { b ->
            b.setOnClickListener { o.moaValue = b.tag as Double; sync() }
        }
        mradButtons.forEach { b ->
            b.setOnClickListener { o.mradValue = b.tag as Double; sync() }
        }
        root.addView(derivedCm)
        root.addView(TextView(a).apply {
            text = a.getString(R.string.optic_advanced)
            setPadding(0, dp(a, 12), 0, 0)
        })
        root.addView(smoaBox)
        smoaBox.setOnCheckedChangeListener { _, _ -> sync() }
        sync()

        AlertDialog.Builder(a)
            .setTitle(if (existing == null) R.string.optic_new else R.string.optic_edit)
            .setView(android.widget.ScrollView(a).apply { addView(root) })
            .setPositiveButton(R.string.save) { _, _ ->
                o.displayName = display.text.toString().trim()
                o.brandModel = brand.text.toString().trim()
                if (o.repr == "CM") {
                    o.cmValue = cmInput.text.toString().replace(',', '.')
                        .toDoubleOrNull() ?: o.cmValue
                }
                o.smoa = smoaBox.isChecked
                if (existing == null) store.addOptic(o) else store.updateOptic(o)
                onDone(o)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    /**
     * Korrigering av ett skuddmerke (musingsUI: blyant per poenglinje):
     * 0,1-steg opp/ned «som en timer-innstilling».
     */
    fun shotEdit(a: Activity, initial: Double, onDone: (Double) -> Unit) {
        var value = initial
        val row = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        val label = TextView(a).apply { textSize = 32f }
        fun render() { label.text = "%.1f".format(value) }
        render()
        row.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "−"; textSize = 26f
            setOnClickListener { value = (value - 0.1).coerceAtLeast(0.0); render() }
        })
        row.addView(label)
        row.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = "+"; textSize = 26f
            setOnClickListener { value = (value + 0.1).coerceAtMost(10.9); render() }
        })
        AlertDialog.Builder(a)
            .setTitle(R.string.correction_title)
            .setView(row)
            .setPositiveButton(R.string.ok) { _, _ -> onDone(value) }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    /** Mikrosamtykke for feilanalysekanalen (per innsending, spec §2). */
    fun failChannelConsent(a: Activity, onAnswer: (Boolean) -> Unit) {
        AlertDialog.Builder(a)
            .setTitle(R.string.fail_channel_title)
            .setMessage(R.string.fail_channel_body)
            .setPositiveButton(R.string.send) { _, _ -> onAnswer(true) }
            .setNegativeButton(R.string.no_thanks) { _, _ -> onAnswer(false) }
            .show()
    }
}
