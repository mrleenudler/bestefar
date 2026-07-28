package no.bestefar.app

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import com.google.android.material.button.MaterialButton

/**
 * Avanserte innstillinger (musingsUI runde 5): våpen, flytt, slett,
 * venstrehåndsmodus og — når DevTools.ENABLED — en Utvikler-meny.
 */
class AvansertActivity : AppCompatActivity() {

    private lateinit var store: Store

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        val content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)

        content.addView(Ui.title(this, getString(R.string.profile_advanced)))

        fun entry(label: String, onClick: () -> Unit) {
            content.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = label
                layoutParams = Ui.matchWrap(4, this@AvansertActivity)
                setOnClickListener { onClick() }
            })
        }

        entry(getString(R.string.profile_weapons_mine)) { weaponsDialog() }
        entry(getString(R.string.profile_move)) {
            Ui.toast(this, R.string.profile_move_todo)
        }
        entry(getString(R.string.profile_delete)) {
            AlertDialog.Builder(this)
                .setMessage(R.string.profile_delete_confirm)
                .setPositiveButton(R.string.profile_delete) { _, _ -> store.wipeAll(); finish() }
                .setNegativeButton(R.string.cancel, null)
                .show()
        }

        // Venstrehåndsmodus (musingsUI runde 5): speiler UI horisontalt
        content.addView(SwitchCompat(this).apply {
            text = getString(R.string.left_handed)
            isChecked = store.leftHanded
            setPadding(Ui.dp(this@AvansertActivity, 4), Ui.dp(this@AvansertActivity, 12),
                0, Ui.dp(this@AvansertActivity, 12))
            setOnCheckedChangeListener { _, on -> store.leftHanded = on; recreate() }
        })
        content.addView(Ui.hint(this, getString(R.string.left_handed_hint)))

        if (DevTools.ENABLED) {
            entry(getString(R.string.dev_menu)) { devMenu() }
        }
    }

    override fun onResume() {
        super.onResume()
        // Speil hele skjermen for venstrehendte (enkel RTL-vending)
        window.decorView.layoutDirection = if (store.leftHanded)
            android.view.View.LAYOUT_DIRECTION_RTL else android.view.View.LAYOUT_DIRECTION_LTR
    }

    private fun devMenu() {
        AlertDialog.Builder(this)
            .setTitle(R.string.dev_menu)
            .setItems(arrayOf(getString(R.string.dev_generate),
                getString(R.string.dev_dummy_scan),
                getString(R.string.dev_always_startup) + ": " +
                    if (store.alwaysShowStartup) "på" else "av")) { _, which ->
                when (which) {
                    0 -> DevTools.generateSeries(this)
                    1 -> DevTools.dummyScan(this)
                    2 -> store.alwaysShowStartup = !store.alwaysShowStartup
                }
            }
            .show()
    }

    private fun weaponsDialog() {
        val root = Ui.col(this, 16)
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.profile_weapons_mine)
            .setView(androidx.core.widget.NestedScrollView(this).apply { addView(root) })
            .setNegativeButton(R.string.close, null)
            .create()
        fun fill() {
            root.removeAllViews()
            store.weapons().forEach { w ->
                val row = Ui.row(this)
                row.addView(android.widget.TextView(this).apply {
                    text = w.shownName; textSize = 16f
                    layoutParams = android.widget.LinearLayout.LayoutParams(0,
                        android.view.ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                })
                row.addView(MaterialButton(this, null,
                    com.google.android.material.R.attr.borderlessButtonStyle).apply {
                    text = getString(R.string.change)
                    setOnClickListener { Dialogs.weaponEdit(this@AvansertActivity, store, w) { fill() } }
                })
                root.addView(row)
            }
            root.addView(MaterialButton(this).apply {
                text = getString(R.string.weapon_add)
                setOnClickListener { Dialogs.weaponEdit(this@AvansertActivity, store, null) { fill() } }
            })
        }
        fill()
        dialog.show()
    }
}
