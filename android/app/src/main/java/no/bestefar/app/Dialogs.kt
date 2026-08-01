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

    /**
     * FORSKNING ER LAGT I BAKGRUNNEN (musingsUI runde 10). Innsamlingen er ikke
     * klar, så alle forskningsdialoger er slått av og bryteren i Avanserte
     * innstillinger er låst av. Koden står urørt bak dette flagget — sett det
     * til true for å slå funksjonaliteten på igjen.
     */
    const val RESEARCH_ENABLED = false

    private fun dp(a: Activity, v: Int) = (v * a.resources.displayMetrics.density).toInt()

    /** «Mitt jaktmål»-rater (musingsUI runde 4). 1 av 13 = nasjonalt snitt. */
    val JAKTMAAL_RATES = listOf(
        0.143 to "1 av 7",
        0.077 to "1 av 13",
        0.05 to "1 av 20",
        0.02 to "1 av 50",
    )

    fun rateLabel(rate: Double): String =
        JAKTMAAL_RATES.minByOrNull { kotlin.math.abs(it.first - rate) }?.second ?: "1 av 13"

    /**
     * «Mitt jaktmål» (musingsUI runde 4): velg akseptabel andel skade-/bomskudd.
     * Tilbys etter tre serier og finnes i Min profil. (i) forklarer fargebruken.
     */
    fun jaktmaalDialog(a: Activity, store: Store, onDone: () -> Unit) {
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        root.addView(TextView(a).apply {
            text = a.getString(R.string.jaktmaal_intro)
            textSize = 15f
        })
        // Egne id-er kreves for at RadioGroup skal deaktivere forrige valg
        // (musingsUI runde 6-bug: uten id fikk vi flere valgte samtidig).
        val group = android.widget.RadioGroup(a)
        var checkId = -1
        JAKTMAAL_RATES.forEach { (rate, label) ->
            val rb = android.widget.RadioButton(a).apply {
                id = android.view.View.generateViewId()
                text = a.getString(R.string.jaktmaal_option, label)
                textSize = 17f
                setOnClickListener { store.rateLimit = rate; store.rateChosen = true }
            }
            group.addView(rb)
            if (store.rateChosen && kotlin.math.abs(store.rateLimit - rate) < rate * 0.1)
                checkId = rb.id
        }
        if (checkId != -1) group.check(checkId)
        root.addView(group)
        // «Hvorfor er noen tall røde?» er flyttet til (i) i Min profil (runde 5)
        AlertDialog.Builder(a)
            .setTitle(R.string.jaktmaal_title)
            .setView(androidx.core.widget.NestedScrollView(a).apply { addView(root) })
            .setPositiveButton(R.string.save) { _, _ ->
                store.rateChosen = true; onDone()
            }
            .setNegativeButton(R.string.cancel) { _, _ -> onDone() }
            .setOnCancelListener { onDone() }
            .show()
    }

    /**
     * Stillingsprompt etter skivescan (musingsUI runde 4): fire stillinger
     * vertikalt med egne ikoner og antall skudd skutt med hver, hjelpemidler
     * (anlegg/reim) horisontalt under som radio-toggler («uten» = deaktivert,
     * klikk aktiv knapp på ny for å slå av). Ingen benk lenger.
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

        var selectedPos = store.currentPosition
        var selectedMod = store.lastModifier(selectedPos)
        val shotsPerPos = store.shotsCountByPosition()

        lateinit var refresh: () -> Unit

        val posCol = LinearLayout(a).apply { orientation = LinearLayout.VERTICAL }
        val modRow = LinearLayout(a).apply { orientation = LinearLayout.HORIZONTAL }
        root.addView(posCol)
        root.addView(TextView(a).apply {
            text = a.getString(R.string.position_modifier)
            setPadding(0, dp(a, 12), 0, dp(a, 4))
        })
        root.addView(modRow)

        refresh = {
            posCol.removeAllViews()
            Position.hoved.forEach { p ->
                val n = shotsPerPos[p] ?: 0
                posCol.addView(positionRow(a, p, n, p == selectedPos) {
                    selectedPos = p
                    selectedMod = store.lastModifier(p)
                    refresh()
                })
            }
            modRow.removeAllViews()
            PosModifier.valgbare.forEach { m ->
                modRow.addView(Ui.choiceButton(a, m.label, selectedMod == m) {
                    // Klikk aktiv -> av (UTEN); ellers velg
                    selectedMod = if (selectedMod == m) PosModifier.UTEN else m
                    refresh()
                }.apply {
                    layoutParams = LinearLayout.LayoutParams(0,
                        ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                        marginEnd = dp(a, 6)
                    }
                })
            }
        }
        refresh()

        root.addView(MaterialButton(a).apply {
            text = a.getString(R.string.ok)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(a, 16) }
            setOnClickListener {
                store.setLastModifier(selectedPos, selectedMod)
                store.currentPosition = selectedPos
                store.currentModifier = selectedMod
                sheet.dismiss()
                onChosen(selectedPos, selectedMod)
            }
        })

        sheet.setContentView(androidx.core.widget.NestedScrollView(a).apply {
            addView(root)
        })
        sheet.setCancelable(false)
        sheet.show()
    }

    /**
     * Stillingsrad i stillingsvelgeren (musingsUI runde 8): ikon via FIT_CENTER-
     * ImageView i en fast-høyde/variabel-bredde-boks, så «Liggende» beholder
     * aspektforholdet (MaterialButton.iconSize tvang kvadrat og klemte den flat).
     * Bredere ikoner (liggende) får være litt lengre; høye (stående) blir smalere.
     */
    private fun positionRow(a: Activity, p: Position, shots: Int, selected: Boolean,
                            onClick: () -> Unit): android.view.View {
        val primary = Ui.themeColor(a, com.google.android.material.R.attr.colorPrimary)
        val onSel = if (selected) android.graphics.Color.WHITE
        else Ui.themeColor(a, android.R.attr.textColorPrimary)
        val row = LinearLayout(a).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = android.graphics.drawable.GradientDrawable().apply {
                cornerRadius = dp(a, 8).toFloat()
                if (selected) setColor(primary)
                else setStroke(dp(a, 1), android.graphics.Color.GRAY)
            }
            isClickable = true
            setOnClickListener { onClick() }
            val padH = dp(a, 12)
            setPadding(padH, dp(a, 6), padH, dp(a, 6))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(a, 6) }
        }
        row.addView(android.widget.ImageView(a).apply {
            setImageResource(p.iconRes)
            scaleType = android.widget.ImageView.ScaleType.FIT_CENTER
            androidx.core.widget.ImageViewCompat.setImageTintList(this,
                android.content.res.ColorStateList.valueOf(onSel))
            layoutParams = LinearLayout.LayoutParams(dp(a, 54), dp(a, 38))
        })
        row.addView(TextView(a).apply {
            text = "  ${p.label}   ($shots skudd)"
            textSize = 16f
            setTextColor(onSel)
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(a, 8) }
        })
        return row
    }

    /** Dagsbekreftelse av våpen — én gang per dag (spec §2). Ammo fjernet i r4. */
    fun weaponDayConfirm(a: Activity, store: Store, onDone: () -> Unit) {
        val ws = store.weapons()
        if (ws.size <= 1 || !store.weaponNeedsDayConfirm()) { onDone(); return }
        val names = ws.map { it.shownName }.toTypedArray()
        var idx = ws.indexOfFirst { it.id == store.selectedWeaponId }.coerceAtLeast(0)
        AlertDialog.Builder(a)
            .setTitle(R.string.weapon_confirm_title)
            .setSingleChoiceItems(names, idx) { _, i -> idx = i }
            .setPositiveButton(R.string.ok) { _, _ ->
                store.selectedWeaponId = ws[idx].id
                store.weaponConfirmedDate = LocalDate.now().toString()
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
        if (!RESEARCH_ENABLED) { onDone(); return }
        val n = store.currentSeasonSeries().size
        val season = Store.seasonKey(System.currentTimeMillis())
        // Aldri samme serie som jaktmål; tidligst 2 serier etter (musingsUI r6)
        val afterJaktmaal = !store.jaktmaalPromptSeen || n >= store.jaktmaalPromptCount + 2
        val due = afterJaktmaal && when (store.consentResearch) {
            "" -> n >= 5
            "senere" -> n - store.consentLastPromptCount >= 10
            // Spør på nytt hver ny sesong, også når man allerede deler (runde 8)
            "ja", "aldri" -> store.researchConsentSeason != season && n >= 1
            else -> false
        }
        if (!due) { onDone(); return }
        store.consentLastPromptCount = n
        AlertDialog.Builder(a)
            .setTitle(R.string.consent_research_title)
            .setMessage(R.string.consent_research_body)
            .setPositiveButton(R.string.consent_yes) { _, _ ->
                store.researchConsentSeason = season; researchConsentYes(a, store) {} }
            .setNegativeButton(R.string.consent_later) { _, _ ->
                store.researchConsentSeason = season; store.consentResearch = "senere" }
            .setNeutralButton(R.string.consent_never) { _, _ ->
                store.researchConsentSeason = season; store.consentResearch = "aldri" }
            .setOnDismissListener { onDone() }
            .show()
    }

    /** «Ja» til forskningssamtykke med 18-årsgate (spec §7). */
    fun researchConsentYes(a: Activity, store: Store, onDone: () -> Unit) {
        val year = LocalDate.now().year
        if (store.birthYear in 1..(year - 18)) {
            store.consentResearch = "ja"
            store.researchConsentSeason = Store.seasonKey(System.currentTimeMillis())
            onDone(); return
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

    /**
     * «Delta i forskning» ved første bruk av jaktloggen (musingsUI runde 3);
     * etterpå fortsetter flyten til Logg jaktskudd uansett svar.
     */
    fun maybeHuntConsent(a: Activity, store: Store, onDone: () -> Unit) {
        if (!RESEARCH_ENABLED) { onDone(); return }
        if (store.consentHunt != "") { onDone(); return }
        researchSharingDialog(a, store, onDone)
    }

    /**
     * Delingsvalg mot forskning (musingsUI runde 3): pitch, låste
     * «deles aldri»-punkter og checkboxer for hva som deles.
     */
    fun researchSharingDialog(a: Activity, store: Store, onDone: () -> Unit) {
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        root.addView(TextView(a).apply {
            text = a.getString(R.string.research_pitch)
            textSize = 15f
        })
        root.addView(TextView(a).apply {
            text = a.getString(R.string.research_never_shared)
            textSize = 15f
            setPadding(0, dp(a, 12), 0, 0)
        })
        // Tomme, låste checkboxer for det som aldri deles
        listOf(R.string.research_never_userid, R.string.research_never_name).forEach {
            root.addView(android.widget.CheckBox(a).apply {
                setText(it); isChecked = false; isEnabled = false
            })
        }
        root.addView(TextView(a).apply {
            text = a.getString(R.string.research_i_share)
            textSize = 15f
            setPadding(0, dp(a, 12), 0, 0)
        })
        val items = listOf("Vilt" to R.string.share_vilt, "Dato" to R.string.share_dato,
            "Posisjon" to R.string.share_posisjon,
            "Skuddsituasjon" to R.string.share_skuddsituasjon)
        // Auto-kryss alt ved første gang — men KUN når forskning er aktivert
        // (musingsUI runde 8); ellers speil valget
        val firstTime = store.consentHunt == "" && store.researchShare.isEmpty() &&
            store.consentResearch == "ja"
        val boxes = items.map { (key, res) ->
            android.widget.CheckBox(a).apply {
                setText(res)
                isChecked = firstTime || key in store.researchShare
                tag = key
            }
        }
        boxes.forEach { root.addView(it) }

        val dialog = AlertDialog.Builder(a)
            .setTitle(R.string.consent_research_join)
            .setView(android.widget.ScrollView(a).apply { addView(root) })
            .setPositiveButton(R.string.save) { _, _ ->
                store.researchShare = boxes.filter { it.isChecked }
                    .map { it.tag as String }.toSet()
                store.consentHunt = "ja"
                onDone()
            }
            .setNegativeButton(R.string.consent_later) { _, _ ->
                store.consentHunt = "senere"; onDone()
            }
            .setNeutralButton(R.string.consent_never) { _, _ ->
                store.consentHunt = "aldri"; onDone()
            }
            .create()
        dialog.setCanceledOnTouchOutside(false)
        dialog.show()
    }

    /**
     * Delingsvalg mot venner (musingsUI runde 4): visningsnavn er perma-delt;
     * øvrige er unchecked default. Avbryt -> «Du må dele visningsnavn …».
     */
    fun friendSharingDialog(a: Activity, store: Store, onDone: () -> Unit = {}) {
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        root.addView(TextView(a).apply {
            text = a.getString(R.string.research_i_share); textSize = 15f
        })
        root.addView(android.widget.CheckBox(a).apply {
            setText(R.string.friend_share_name); isChecked = true; isEnabled = false
        })
        val items = listOf(
            "Skudd" to R.string.friend_share_shots,
            "Score" to R.string.friend_share_score,
            "Utvikling" to R.string.friend_share_trend,
            "Fellinger" to R.string.friend_share_kills,
            "Telefon" to R.string.friend_share_phone,
            "Lag" to R.string.friend_share_teams,
            "Hjemkommune" to R.string.friend_share_kommune)
        val boxes = items.map { (key, res) ->
            android.widget.CheckBox(a).apply {
                setText(res); isChecked = key in store.friendShare; tag = key
            }
        }
        boxes.forEach { root.addView(it) }
        val dialog = AlertDialog.Builder(a)
            .setTitle(R.string.sharing_friends)
            .setView(androidx.core.widget.NestedScrollView(a).apply { addView(root) })
            .setPositiveButton(R.string.save) { _, _ ->
                store.friendShare = boxes.filter { it.isChecked }
                    .map { it.tag as String }.toSet() + "Navn"
                store.friendShareActive = true
                onDone()
            }
            .setNegativeButton(R.string.cancel) { _, _ ->
                AlertDialog.Builder(a)
                    .setMessage(R.string.friends_need_display_name_view)
                    .setPositiveButton(R.string.ok, null).show()
            }
            .create()
        dialog.setCanceledOnTouchOutside(false)
        dialog.show()
    }

    /**
     * Legg til / endre våpen (musingsUI runde 4): visningsnavn, ikonvalg og
     * våpenmodell. Optikk og ammunisjon er FJERNET for å forenkle
     * brukeropplevelsen (behovet vurdert som marginalt); se OpticProfile /
     * KalkulatorActivity for utkommentert referanse.
     */
    fun weaponEdit(a: Activity, store: Store, existing: Weapon?, onDone: () -> Unit) {
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(a, 24), dp(a, 8), dp(a, 24), 0)
        }
        val display = EditText(a).apply {
            hint = a.getString(R.string.weapon_display_hint)
        }
        Ui.capitalize(display)
        display.setText(existing?.displayName ?: "")
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
            hint = a.getString(R.string.weapon_model_hint)
        }
        Ui.capitalize(name)
        name.setText(existing?.name ?: "")
        root.addView(name)

        val dialog = AlertDialog.Builder(a)
            .setTitle(if (existing == null) R.string.weapon_add else R.string.weapon_edit)
            .setView(androidx.core.widget.NestedScrollView(a).apply { addView(root) })
            .setPositiveButton(R.string.save) { _, _ ->
                val n = name.text.toString().trim()
                val d = display.text.toString().trim()
                if (n.isEmpty() && d.isEmpty()) return@setPositiveButton
                val savedId: String
                if (existing == null) {
                    val w = Weapon(Store.newId(), n, null, false, "",
                        displayName = d, icon = iconKey)
                    store.addWeapon(w)
                    savedId = w.id
                } else {
                    existing.name = n
                    existing.displayName = d
                    existing.icon = iconKey
                    store.updateWeapon(existing)
                    savedId = existing.id
                }
                // Lagret våpen blir valgt våpen (musingsUI runde 3)
                store.selectedWeaponId = savedId
                onDone()
            }
            .setNegativeButton(R.string.cancel, null)
            .create()
        // Klikk utenfor skal ikke lukke; kun Lagre/Avbryt/tilbake (runde 3)
        dialog.setCanceledOnTouchOutside(false)
        dialog.show()
    }

    /*
     * OPTIKK-FUNKSJONALITET FJERNET (musingsUI runde 4).
     * opticEdit(), opticChooser(), resolveOptic() og clickValueLabel() er tatt
     * ut sammen med optikk-kalkulatoren for å forenkle brukeropplevelsen.
     * OpticProfile-modellen og Store.optics()/clickCmFor() beholdes utkommentert
     * i bruk, slik at funksjonaliteten kan gjeninnføres uten datatap.
     */

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
