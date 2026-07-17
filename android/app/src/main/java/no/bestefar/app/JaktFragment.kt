package no.bestefar.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import androidx.appcompat.widget.SwitchCompat
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Jakt-flaten (spec §4): valgfritt dagsoppsett (arter, stedslogging) og
 * ett-trykks hurtiglogg. Jaktsamtykke tilbys ved første bruk (spec §7).
 */
class JaktFragment : RebuildFragment() {

    override fun rebuild() {
        val a = requireActivity()
        val store = Store.get(a)
        content.removeAllViews()
        content.addView(Ui.title(a, getString(R.string.tab_jakt)))

        content.addView(SwitchCompat(a).apply {
            text = getString(R.string.hunt_mode)
            isChecked = store.huntMode
            setOnCheckedChangeListener { _, on -> store.huntMode = on }
        })

        content.addView(Ui.section(a, getString(R.string.hunt_day_setup)))
        val chips = ChipGroup(a)
        Species.entries.filter { it != Species.ANNET }.forEach { s ->
            chips.addView(Chip(a).apply {
                text = s.label; isCheckable = true
                isChecked = s.name in store.huntSpecies
                setOnClickListener {
                    store.huntSpecies =
                        if (isChecked) store.huntSpecies + s.name
                        else store.huntSpecies - s.name
                }
            })
        }
        content.addView(chips)

        content.addView(SwitchCompat(a).apply {
            text = getString(R.string.hunt_location)
            isChecked = store.locationLogging
            setOnCheckedChangeListener { _, on ->
                store.locationLogging = on
                if (on && ContextCompat.checkSelfPermission(a,
                        Manifest.permission.ACCESS_FINE_LOCATION)
                    != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(a,
                        arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), 2)
                }
            }
        })

        content.addView(MaterialButton(a).apply {
            text = getString(R.string.hunt_log_button)
            textSize = 18f
            layoutParams = Ui.matchWrap(16, a).apply { height = Ui.dp(a, 64) }
            setOnClickListener {
                Dialogs.maybeHuntConsent(a, store) {
                    startActivity(Intent(a, HuntLogActivity::class.java))
                }
            }
        })

        content.addView(Ui.section(a, getString(R.string.hunt_recent)))
        val hunts = store.allHunts().sortedByDescending { it.ts }
        if (hunts.isEmpty()) {
            content.addView(Ui.hint(a, getString(R.string.hunt_none)))
        } else {
            val fmt = DateTimeFormatter.ofPattern("d.M.yy HH:mm")
            hunts.take(8).forEach { h ->
                val t = Instant.ofEpochMilli(h.ts).atZone(ZoneId.systemDefault()).format(fmt)
                val follow = h.followUp?.let { " → ${it.label}" } ?: ""
                val moving = if (h.moving) " · i bevegelse" else ""
                content.addView(Ui.body(a,
                    "$t · ${h.species.label} · ${h.distanceM} m · ${h.angle.label}$moving · " +
                    h.outcome.label + follow))
            }
        }
    }
}
