package no.bestefar.app

import android.content.Intent
import android.text.Editable
import android.text.InputFilter
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioButton
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.widget.SwitchCompat
import com.google.android.material.button.MaterialButton
import java.time.LocalDate

/**
 * Dropdown-paneler for våpen, avstand, jakt, stilling og meny (musingsUI):
 * dekker bare deler av skjermen; trykk på samme knapp igjen lukker.
 */
object Panels {

    fun build(i: Int, a: MainActivity, refresh: () -> Unit): View = when (i) {
        0 -> vapen(a, refresh)
        1 -> avstand(a, refresh)
        2 -> jakt(a)
        3 -> stilling(a, refresh)
        else -> meny(a)
    }

    // ---------- Våpen: kun visningsnavn + ikon; ingen ammo/forklaring ----------

    private fun vapen(a: MainActivity, refresh: () -> Unit): View {
        val store = a.store
        val col = Ui.col(a, 12)
        col.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = a.getString(R.string.weapon_add)
            setOnClickListener { Dialogs.weaponEdit(a, store, null) { refresh() } }
        })
        val selectedId = store.selectedWeapon()?.id
        store.weapons().forEach { w ->
            val row = Ui.row(a)
            row.addView(ImageView(a).apply {
                setImageResource(WeaponIcons.res(w.icon))
                adjustViewBounds = true
                layoutParams = LinearLayout.LayoutParams(Ui.dp(a, 44), Ui.dp(a, 28))
                contentDescription = w.shownName
            })
            row.addView(RadioButton(a).apply {
                textSize = 17f
                text = w.shownName
                isChecked = w.id == selectedId
                setPadding(Ui.dp(a, 8), Ui.dp(a, 8), 0, Ui.dp(a, 8))
                setOnClickListener {
                    store.selectedWeaponId = w.id
                    store.weaponConfirmedDate = LocalDate.now().toString()
                    a.closeDropdown()
                }
            })
            col.addView(row)
        }
        return col
    }

    // ---------- Avstand: vertikale valg + «X m»-knapp (musingsUI) ----------

    private fun avstand(a: MainActivity, refresh: () -> Unit): View {
        val store = a.store
        val col = Ui.col(a, 12)
        var editing = false

        fun distButton(d: Int, label: String = "$d m", onClick: () -> Unit) =
            MaterialButton(a, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = label
                alpha = if (store.distanceM == d) 1f else 0.55f
                layoutParams = Ui.matchWrap(4, a)
                setOnClickListener { onClick() }
            }

        listOf(50, 100, 150, 200, 300).forEach { d ->
            col.addView(distButton(d) { store.distanceM = d; a.closeDropdown() })
        }

        val custom = store.customDistance
        val customLabel = if (custom > 0) "$custom m" else a.getString(R.string.distance_x)
        val inputRow = Ui.row(a).apply { visibility = View.GONE }
        val input = EditText(a).apply {
            hint = a.getString(R.string.distance_custom_hint)
            inputType = InputType.TYPE_CLASS_NUMBER
            filters = arrayOf(InputFilter.LengthFilter(4))
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        inputRow.addView(input)
        inputRow.addView(MaterialButton(a).apply {
            text = a.getString(R.string.distance_set)
            setOnClickListener {
                val v = input.text.toString().toIntOrNull()
                if (v != null && v in 10..9999) {
                    store.customDistance = v
                    store.distanceM = v
                    a.closeDropdown()
                }
            }
        })

        col.addView(distButton(custom.takeIf { it > 0 } ?: -1, customLabel) {
            when {
                custom <= 0 -> { editing = true; inputRow.visibility = View.VISIBLE }
                store.distanceM != custom -> { store.distanceM = custom; a.closeDropdown() }
                // Allerede valgt -> trykk på ny åpner tekstfeltet igjen
                else -> { editing = !editing
                    inputRow.visibility = if (editing) View.VISIBLE else View.GONE }
            }
        })
        col.addView(inputRow)
        return col
    }

    // ---------- Jakt: bare Logg jaktskudd (musingsUI) ----------

    private fun jakt(a: MainActivity): View {
        val col = Ui.col(a, 12)
        col.addView(MaterialButton(a).apply {
            text = a.getString(R.string.hunt_log_button)
            textSize = 18f
            layoutParams = Ui.matchWrap(0, a).apply { height = Ui.dp(a, 64) }
            setOnClickListener {
                a.closeDropdown()
                Dialogs.maybeHuntConsent(a, a.store) {
                    a.startActivity(Intent(a, HuntLogActivity::class.java))
                }
            }
        })
        return col
    }

    // ---------- Stilling: stillinger venstre, støtte til høyre ----------

    private fun stilling(a: MainActivity, refresh: () -> Unit): View {
        val store = a.store
        val col = Ui.col(a, 12)
        val row = LinearLayout(a).apply { orientation = LinearLayout.HORIZONTAL }

        val posCol = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 0.55f)
        }
        (Position.hoved + Position.BENK).forEach { p ->
            posCol.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = p.label
                alpha = if (p == store.currentPosition) 1f else 0.55f
                layoutParams = Ui.matchWrap(4, a)
                setOnClickListener {
                    store.currentPosition = p
                    store.currentModifier = store.lastModifier(p)
                    refresh()
                }
            })
        }
        row.addView(posCol)

        val modCol = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 0.45f).apply {
                leftMargin = Ui.dp(a, 8)
            }
        }
        if (store.currentPosition != Position.BENK) {
            PosModifier.entries.forEach { m ->
                modCol.addView(MaterialButton(a, null,
                    com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                    text = m.label
                    alpha = if (store.currentModifier == m) 1f else 0.55f
                    layoutParams = Ui.matchWrap(4, a)
                    setOnClickListener {
                        store.currentModifier = m
                        store.setLastModifier(store.currentPosition, m)
                        refresh()
                    }
                })
            }
        }
        row.addView(modCol)
        col.addView(row)

        // «Spør etter hver serie» — default på (musingsUI); av = manuell stilling
        col.addView(SwitchCompat(a).apply {
            text = a.getString(R.string.stilling_ask)
            isChecked = !store.manualPosition
            setOnCheckedChangeListener { _, on -> store.manualPosition = !on }
        })
        return col
    }

    // ---------- Meny: mindre knapper inntil høyre side (musingsUI) ----------

    private fun meny(a: MainActivity): View {
        val store = a.store
        val col = Ui.col(a, 12)

        fun entry(label: String, onClick: () -> Unit) {
            col.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = label
                textSize = 14f
                layoutParams = Ui.matchWrap(4, a).apply { height = Ui.dp(a, 44) }
                setOnClickListener { a.closeDropdown(); onClick() }
            })
        }

        entry(a.getString(R.string.menu_profile)) {
            a.startActivity(Intent(a, ProfilActivity::class.java))
        }
        entry(a.getString(R.string.weapon_edit)) { editWeaponChooser(a) }
        entry(a.getString(R.string.menu_series)) {
            a.startActivity(Intent(a, SerieloggActivity::class.java))
        }
        entry(a.getString(R.string.kalkulator)) {
            a.startActivity(Intent(a, KalkulatorActivity::class.java))
        }
        entry(a.getString(R.string.menu_advanced_stats)) { statsDialog(a) }
        entry(a.getString(R.string.menu_about)) { aboutDialog(a) }
        entry(a.getString(R.string.menu_tutorial)) { a.showHome(); a.showTutorial() }
        entry(a.getString(R.string.search)) { searchDialog(a) }
        return col
    }

    /** «Legg til våpen» er et valg under Rediger våpen (musingsUI). */
    private fun editWeaponChooser(a: MainActivity) {
        val store = a.store
        val ws = store.weapons()
        val items = ws.map { it.shownName } + a.getString(R.string.weapon_add)
        AlertDialog.Builder(a)
            .setTitle(R.string.weapon_edit)
            .setItems(items.toTypedArray()) { _, i ->
                if (i < ws.size) Dialogs.weaponEdit(a, store, ws[i]) {}
                else Dialogs.weaponEdit(a, store, null) {}
            }.show()
    }

    private fun statsDialog(a: MainActivity) {
        val store = a.store
        // Tekniske mål KUN her og i forskningseksport (spec §1)
        val evid = store.currentSeasonSeries().filter { it.countsInEvidence }
        val sigma = Stats.sigmaCmAt100(evid)
        val msg = if (sigma == null) "Ingen data denne sesongen."
        else "σ (100 m-ekvivalent): %.1f cm\nR95: %.1f cm\nSpredning: %.2f MOA\n\nMålt om siktepunktet (%d skudd). Benk holdes utenfor."
            .format(sigma, Stats.r95Cm(sigma), Stats.moa(sigma), Stats.shotCount(evid))
        AlertDialog.Builder(a).setTitle(R.string.menu_advanced_stats)
            .setMessage(msg).setPositiveButton(R.string.ok, null).show()
    }

    private fun aboutDialog(a: MainActivity) {
        AlertDialog.Builder(a).setTitle(R.string.menu_about)
            .setMessage(a.getString(R.string.help_body) +
                "\n\nVersjon " + a.packageManager
                    .getPackageInfo(a.packageName, 0).versionName)
            .setPositiveButton(R.string.ok, null).show()
    }

    // ---------- Søk (musingsUI): fritekst mot funksjons-/info-indeks ----------

    private data class SearchEntry(val title: String, val keywords: String,
                                   val action: () -> Unit)

    private fun searchIndex(a: MainActivity): List<SearchEntry> {
        val store = a.store
        return listOf(
            SearchEntry(a.getString(R.string.menu_series),
                "serie serier logg historikk resultat slette gjennomsnitt") {
                a.startActivity(Intent(a, SerieloggActivity::class.java)) },
            SearchEntry(a.getString(R.string.kalkulator),
                "optikk kalkulator moa mrad cm klikk omregning smoa") {
                a.startActivity(Intent(a, KalkulatorActivity::class.java)) },
            SearchEntry(a.getString(R.string.weapon_edit),
                "våpen rediger legg til optikk ammo ammunisjon kikkert klikkverdi") {
                editWeaponChooser(a) },
            SearchEntry(a.getString(R.string.menu_profile),
                "profil samtykke skadeskytingsrate sletting data fødselsår lag") {
                a.startActivity(Intent(a, ProfilActivity::class.java)) },
            SearchEntry(a.getString(R.string.tab_innsikt),
                "innsikt kompetanse kart forsvarlig hold frekvens statistikk") {
                a.select(4) },
            SearchEntry(a.getString(R.string.hunt_log_button),
                "jakt jaktlogg hurtiglogg skudd art hold vinkling utfall ettersøk") {
                Dialogs.maybeHuntConsent(a, store) {
                    a.startActivity(Intent(a, HuntLogActivity::class.java)) } },
            SearchEntry(a.getString(R.string.tab_avstand),
                "avstand meter hold egendefinert") { a.select(1) },
            SearchEntry(a.getString(R.string.tab_stilling),
                "stilling liggende sittende knestående stående benk anlegg reim") {
                a.select(3) },
            SearchEntry(a.getString(R.string.menu_advanced_stats),
                "statistikk avansert sigma r95 moa spredning teknisk") { statsDialog(a) },
            SearchEntry(a.getString(R.string.menu_about),
                "om appen hjelp ikon forklaring skadeskytingsrate forskning versjon") {
                aboutDialog(a) },
            SearchEntry(a.getString(R.string.menu_tutorial),
                "tutorial gjennomgang intro opplæring velkommen") {
                a.showHome(); a.showTutorial() },
        )
    }

    private fun searchDialog(a: MainActivity) {
        val root = LinearLayout(a).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(Ui.dp(a, 20), Ui.dp(a, 8), Ui.dp(a, 20), Ui.dp(a, 8))
        }
        val input = EditText(a).apply { hint = a.getString(R.string.search_hint) }
        val results = LinearLayout(a).apply { orientation = LinearLayout.VERTICAL }
        root.addView(input); root.addView(results)
        val dialog = AlertDialog.Builder(a)
            .setTitle(R.string.search)
            .setView(root)
            .setNegativeButton(R.string.cancel, null)
            .create()
        val index = searchIndex(a)
        fun refresh(q: String) {
            results.removeAllViews()
            val terms = q.lowercase().split(" ").filter { it.isNotBlank() }
            val hits = if (terms.isEmpty()) emptyList()
            else index.filter { e ->
                terms.all { t ->
                    e.title.lowercase().contains(t) || e.keywords.contains(t)
                }
            }
            hits.take(6).forEach { e ->
                results.addView(MaterialButton(a, null,
                    com.google.android.material.R.attr.borderlessButtonStyle).apply {
                    text = e.title
                    setOnClickListener { dialog.dismiss(); e.action() }
                })
            }
        }
        input.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, x: Int, y: Int, z: Int) {}
            override fun onTextChanged(s: CharSequence?, x: Int, y: Int, z: Int) {}
            override fun afterTextChanged(s: Editable?) = refresh(s?.toString() ?: "")
        })
        dialog.show()
    }
}
