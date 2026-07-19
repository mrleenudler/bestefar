package no.bestefar.app

import android.os.Bundle
import android.text.InputType
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import androidx.appcompat.widget.SwitchCompat
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import java.time.LocalDate

/** Felles skjelett for de programmatiske fanene. */
abstract class RebuildFragment : Fragment() {
    protected lateinit var content: LinearLayout

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?,
                              savedInstanceState: Bundle?): View {
        content = Ui.col(requireContext())
        return Ui.scroll(requireContext(), content)
    }

    override fun onResume() {
        super.onResume()
        rebuild()
    }

    protected abstract fun rebuild()
}

/**
 * Våpen-fanen (spec §2): forhåndsvalgt og synlig som valg; med bare ett
 * registrert våpen velges det automatisk uten prompt.
 */
class VapenFragment : RebuildFragment() {
    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()
        content.addView(Ui.title(a, getString(R.string.tab_vapen)))

        // «Legg til våpen» øverst (musingsUI); finnes også i Meny
        content.addView(MaterialButton(a, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.weapon_add)
            layoutParams = Ui.matchWrap(4, a)
            setOnClickListener { Dialogs.weaponEdit(a, store, null, false) { rebuild() } }
        })

        val ws = store.weapons()
        if (ws.isEmpty()) {
            content.addView(Ui.body(a, getString(R.string.weapon_none)))
        } else {
            val selectedId = store.selectedWeapon()?.id
            ws.forEach { w ->
                val row = Ui.row(a)
                row.addView(android.widget.ImageView(a).apply {
                    setImageResource(WeaponIcons.res(w.icon))
                    adjustViewBounds = true
                    layoutParams = LinearLayout.LayoutParams(
                        Ui.dp(a, 44), Ui.dp(a, 28))
                    contentDescription = w.shownName
                })
                row.addView(RadioButton(a).apply {
                    textSize = 17f
                    text = buildString {
                        append(w.shownName)
                        if (w.ammoSplit && w.ammoName.isNotBlank()) append(" — ${w.ammoName}")
                    }
                    isChecked = w.id == selectedId
                    setPadding(Ui.dp(a, 8), Ui.dp(a, 8), 0, Ui.dp(a, 8))
                    setOnClickListener {
                        store.selectedWeaponId = w.id
                        store.weaponConfirmedDate = LocalDate.now().toString()
                        rebuild()
                    }
                })
                content.addView(row)
            }
        }
        content.addView(Ui.hint(a, getString(R.string.weapon_manage_hint)))
    }
}

/** Avstand-fanen: 100 m som standard, enkelt å endre (spec §1/§2). */
class AvstandFragment : RebuildFragment() {
    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()
        content.addView(Ui.title(a, getString(R.string.distance_title)))
        content.addView(Ui.body(a, getString(R.string.distance_current, store.distanceM)))

        val group = ChipGroup(a).apply { isSingleSelection = true }
        listOf(50, 100, 150, 200, 300).forEach { d ->
            group.addView(Chip(a).apply {
                text = "$d m"; isCheckable = true
                isChecked = store.distanceM == d
                setOnClickListener { store.distanceM = d; rebuild() }
            })
        }
        content.addView(group)

        val row = Ui.row(a)
        val input = EditText(a).apply {
            hint = getString(R.string.distance_custom_hint)
            inputType = InputType.TYPE_CLASS_NUMBER
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        row.addView(input)
        row.addView(MaterialButton(a).apply {
            text = getString(R.string.distance_set)
            setOnClickListener {
                input.text.toString().toIntOrNull()?.let {
                    if (it in 10..1000) { store.distanceM = it; rebuild() }
                }
            }
        })
        content.addView(row)
    }
}

/**
 * Stilling-fanen: fire hovedstillinger + benk, modifikator-chips (siste
 * huskes per stilling), og «ikke spør — manuell stilling» (spec §2).
 */
class StillingFragment : RebuildFragment() {
    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()
        content.addView(Ui.title(a, getString(R.string.tab_stilling)))
        content.addView(Ui.body(a, getString(R.string.stilling_current,
            store.currentPosition.label, store.currentModifier.label)))

        store.practicePosition?.let { p ->
            content.addView(Ui.body(a, "Øvelsesmodus: ${p.label} er forhåndsvalgt."))
            content.addView(MaterialButton(a, null,
                com.google.android.material.R.attr.borderlessButtonStyle).apply {
                text = "Avslutt øvelse"
                setOnClickListener { store.practicePosition = null; rebuild() }
            })
        }

        (Position.hoved + Position.BENK).forEach { p ->
            content.addView(MaterialButton(a).apply {
                text = p.label
                alpha = if (p == store.currentPosition) 1f else 0.55f
                layoutParams = Ui.matchWrap(8, a)
                setOnClickListener {
                    store.currentPosition = p
                    store.currentModifier = store.lastModifier(p)
                    rebuild()
                }
            })
        }

        if (store.currentPosition != Position.BENK) {
            content.addView(Ui.body(a, getString(R.string.position_modifier)))
            val group = ChipGroup(a).apply { isSingleSelection = true }
            PosModifier.entries.forEach { m ->
                group.addView(Chip(a).apply {
                    text = m.label; isCheckable = true
                    isChecked = store.currentModifier == m
                    setOnClickListener {
                        store.currentModifier = m
                        store.setLastModifier(store.currentPosition, m)
                        rebuild()
                    }
                })
            }
            content.addView(group)
        }

        content.addView(Ui.vspace(a, 12))
        content.addView(SwitchCompat(a).apply {
            text = getString(R.string.stilling_manual)
            isChecked = store.manualPosition
            setOnCheckedChangeListener { _, on -> store.manualPosition = on }
        })
        content.addView(Ui.hint(a, getString(R.string.stilling_manual_hint)))
    }
}
