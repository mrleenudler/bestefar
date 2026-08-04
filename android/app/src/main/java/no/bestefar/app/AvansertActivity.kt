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
    private var queueBtn: MaterialButton? = null
    private var queueHint: android.widget.TextView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        val content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)

        // Tittelrad med equalizer-ikonet (musingsUI runde 12). Her er ikonet
        // sidens identitet og IKKE klikkbart — vi er allerede framme.
        val titleRow = Ui.row(this)
        titleRow.addView(android.widget.ImageView(this).apply {
            setImageResource(R.drawable.ic_settings_sliders)
            scaleType = android.widget.ImageView.ScaleType.FIT_CENTER
            androidx.core.widget.ImageViewCompat.setImageTintList(this,
                android.content.res.ColorStateList.valueOf(Ui.themeColor(
                    this@AvansertActivity,
                    com.google.android.material.R.attr.colorPrimary)))
            layoutParams = android.widget.LinearLayout.LayoutParams(
                Ui.dp(this@AvansertActivity, 28), Ui.dp(this@AvansertActivity, 28))
                .apply { marginEnd = Ui.dp(this@AvansertActivity, 10) }
        })
        titleRow.addView(Ui.title(this, getString(R.string.profile_advanced)))
        content.addView(titleRow)

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
            Ui.warningDialog(this)
                .setMessage(R.string.profile_delete_confirm)
                .setPositiveButton(R.string.profile_delete) { _, _ -> store.wipeAll(); finish() }
                .setNegativeButton(R.string.cancel, null)
                .show()
        }

        // «Fjern inaktiv lagleder» (musingsUI runde 6/9) — FRONT-END-SKJELETT.
        entry(getString(R.string.team_remove_inactive)) { removeInactiveLeader() }

        // Sikkerhetskopi (backend_spec §2, musingsUI runde 12)
        entry(getString(R.string.backup_title)) { backupMenu() }

        // Del med forskning av/på (musingsUI runde 8): valget bor nå her, ikke i
        // en oppstartspopup. På = "ja", av = "aldri". LÅST AV i runde 10 —
        // forskning er lagt i bakgrunnen til innsamlingen er klar.
        content.addView(SwitchCompat(this).apply {
            text = getString(R.string.research_share_toggle)
            isChecked = Dialogs.RESEARCH_ENABLED && store.consentResearch == "ja"
            isEnabled = Dialogs.RESEARCH_ENABLED
            setPadding(Ui.dp(this@AvansertActivity, 4), Ui.dp(this@AvansertActivity, 12),
                0, Ui.dp(this@AvansertActivity, 12))
            setOnCheckedChangeListener { _, on ->
                if (on) {
                    // 18-årsgate før forskning aktiveres (spec §7)
                    Dialogs.researchConsentYes(this@AvansertActivity, store) {
                        if (store.consentResearch == "ja")
                            store.researchConsentSeason = Store.seasonKey(System.currentTimeMillis())
                        else isChecked = false   // avslått (alder) -> tilbakestill
                    }
                } else store.consentResearch = "aldri"
            }
        })
        content.addView(Ui.hint(this, getString(
            if (Dialogs.RESEARCH_ENABLED) R.string.research_share_toggle_hint
            else R.string.research_share_paused)))

        // Del bilder med utvikler av/på (musingsUI runde 8): flyttet fra
        // oppstartspopup til her. På = "ja", av = "nei".
        content.addView(SwitchCompat(this).apply {
            text = getString(R.string.share_dev_images)
            isChecked = store.shareDevImagesActive
            setPadding(Ui.dp(this@AvansertActivity, 4), Ui.dp(this@AvansertActivity, 12),
                0, Ui.dp(this@AvansertActivity, 12))
            setOnCheckedChangeListener { _, on -> store.shareDevImages = if (on) "ja" else "nei" }
        })
        content.addView(Ui.hint(this, getString(R.string.share_dev_images_hint)))

        // Kun wifi (backend_spec §6): køen er fullskala-JPEG-er, og de skal
        // ikke spise mobildata uten at brukeren har bedt om det.
        content.addView(SwitchCompat(this).apply {
            text = getString(R.string.upload_wifi_only)
            isChecked = store.uploadWifiOnly
            setPadding(Ui.dp(this@AvansertActivity, 4), Ui.dp(this@AvansertActivity, 12),
                0, Ui.dp(this@AvansertActivity, 12))
            setOnCheckedChangeListener { _, on -> store.uploadWifiOnly = on }
        })
        content.addView(Ui.hint(this, getString(R.string.upload_wifi_only_hint)))

        // «Send nå» viser hva som faktisk står i kø — tallet var før en teller
        // som bare kunne vokse, siden ingenting sendte noe.
        val btn = MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            layoutParams = Ui.matchWrap(4, this@AvansertActivity)
            setOnClickListener { sendNow() }
        }
        val hint = Ui.hint(this, "")
        queueBtn = btn
        queueHint = hint
        content.addView(btn)
        content.addView(hint)
        updateQueueUi()

        // Lagre scannede skjermbilder i bildearkivet (musingsUI runde 10/12).
        // Tre valg, ikke av/på: «alle» fyller kamerarullen med treningsserier,
        // «aldri» kaster bildene man faktisk ville beholdt.
        content.addView(android.widget.TextView(this).apply {
            text = getString(R.string.save_scans_gallery_choice)
            setPadding(Ui.dp(this@AvansertActivity, 4), Ui.dp(this@AvansertActivity, 12),
                0, Ui.dp(this@AvansertActivity, 4))
        })
        val galleryRow = Ui.row(this)
        val galleryModes = listOf(
            Store.GallerySave.ALDRI to R.string.save_scans_never,
            Store.GallerySave.ALLE to R.string.save_scans_all,
            Store.GallerySave.BESTE to R.string.save_scans_best)
        lateinit var fillGallery: () -> Unit
        fillGallery = {
            galleryRow.removeAllViews()
            galleryModes.forEach { (mode, res) ->
                galleryRow.addView(Ui.choiceButton(this, getString(res),
                    store.saveScansMode == mode, small = true) {
                    store.saveScansMode = mode
                    fillGallery()
                }.apply {
                    layoutParams = android.widget.LinearLayout.LayoutParams(0,
                        android.view.ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                        marginEnd = Ui.dp(this@AvansertActivity, 6)
                    }
                })
            }
        }
        fillGallery()
        content.addView(galleryRow, Ui.matchWrap(0, this))
        content.addView(Ui.hint(this, getString(R.string.save_scans_gallery_hint)))
        content.addView(Ui.hint(this, getString(R.string.save_scans_best_hint)))

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
        updateQueueUi()
    }

    // ---------- Opplastingskø (backend_spec §6) ----------

    private fun updateQueueUi() {
        val n = Sync.pending(this)
        queueBtn?.apply {
            text = getString(R.string.upload_send_now)
            isEnabled = n > 0
        }
        queueHint?.text = when {
            n == 0 && store.lastSyncTs > 0L -> getString(R.string.upload_queue_empty_since,
                java.time.Instant.ofEpochMilli(store.lastSyncTs)
                    .atZone(java.time.ZoneId.systemDefault())
                    .format(ResultActivity.DATE_TIME_FMT))
            n == 0 -> getString(R.string.upload_queue_empty)
            else -> getString(R.string.upload_queue_count, n)
        }
    }

    private fun sendNow() {
        queueBtn?.apply { isEnabled = false; text = getString(R.string.upload_sending) }
        // force = true: brukeren ba om det uttrykkelig, så kun-wifi settes til
        // side. «Er vi på nett i det hele tatt» gjelder fortsatt.
        Sync.flush(this, force = true) { o ->
            updateQueueUi()
            Ui.toast(this, when {
                o.skipped -> getString(R.string.upload_offline)
                o.tried == 0 -> getString(R.string.upload_queue_empty)
                o.failed > 0 -> getString(R.string.upload_partial, o.sent, o.failed)
                o.dropped > 0 -> getString(R.string.upload_done_dropped, o.sent, o.dropped)
                else -> getString(R.string.upload_done, o.sent)
            })
        }
    }

    // ---------- Sikkerhetskopi (backend_spec §2) ----------

    private fun backupMenu() {
        AlertDialog.Builder(this)
            .setTitle(R.string.backup_title)
            .setItems(arrayOf(getString(R.string.backup_show_code),
                getString(R.string.backup_now),
                getString(R.string.backup_restore))) { _, which ->
                when (which) {
                    0 -> showRecoveryCode()
                    1 -> ensureCode { doBackup(force = false) }
                    2 -> confirmRestore()
                }
            }
            .setNegativeButton(R.string.close, null)
            .show()
    }

    /**
     * Koden lages ved første behov og vises ÉN gang med en tydelig beskjed om
     * hva den er til. Den kan hentes fram igjen herfra — alternativet ville
     * vært en kode brukeren aldri kan finne igjen, og da tar ingen kopi.
     */
    private fun ensureCode(after: () -> Unit) {
        if (store.backupCode.isEmpty()) store.backupCode = Backup.newRecoveryCode()
        if (store.backupCodeShown) { after(); return }
        showRecoveryCode(after)
    }

    private fun showRecoveryCode(after: (() -> Unit)? = null) {
        if (store.backupCode.isEmpty()) store.backupCode = Backup.newRecoveryCode()
        val col = Ui.col(this, 24)
        col.addView(android.widget.TextView(this).apply {
            text = getString(R.string.backup_code_body)
            textSize = 15f
        })
        col.addView(android.widget.TextView(this).apply {
            text = Backup.formatCode(store.backupCode)
            textSize = 22f
            typeface = android.graphics.Typeface.MONOSPACE
            gravity = android.view.Gravity.CENTER
            setTextIsSelectable(true)          // så koden kan kopieres ut
            setPadding(0, Ui.dp(this@AvansertActivity, 20), 0, 0)
        })
        AlertDialog.Builder(this)
            .setTitle(R.string.backup_code_title)
            .setView(androidx.core.widget.NestedScrollView(this).apply { addView(col) })
            .setPositiveButton(R.string.backup_code_saved) { _, _ ->
                store.backupCodeShown = true
                after?.invoke()
            }
            .setCancelable(false)
            .show()
    }

    private fun doBackup(force: Boolean) {
        Ui.toast(this, R.string.backup_working)
        Api.io {
            val resp = Backup.upload(this, store.backupCode, force)
            Api.ui {
                when {
                    resp.ok -> Ui.toast(this, getString(R.string.backup_done,
                        resp.body.length.coerceAtLeast(1)))
                    resp.code == 401 -> Ui.toast(this, R.string.backup_need_login)
                    // 409: serveren har en NYERE kopi. Overskriving er et
                    // bevisst valg («gjenopprett fra denne enheten»), ikke noe
                    // vi skal gjøre automatisk.
                    resp.code == 409 -> Ui.warningDialog(this)
                        .setMessage(R.string.backup_conflict)
                        .setPositiveButton(R.string.overwrite) { _, _ -> doBackup(force = true) }
                        .setNegativeButton(R.string.cancel, null)
                        .show()
                    resp.code == 0 -> Ui.toast(this, R.string.backup_offline)
                    else -> Ui.toast(this, getString(R.string.backup_failed, resp.code))
                }
            }
        }
    }

    private fun confirmRestore() {
        Ui.warningDialog(this)
            .setMessage(R.string.backup_restore_confirm)
            .setPositiveButton(R.string.backup_restore) { _, _ -> doRestore() }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun doRestore() {
        Ui.toast(this, R.string.backup_working)
        Api.io {
            val result = try {
                Backup.downloadAndRestore(this, store.backupCode) to null
            } catch (e: Backup.BadCodeException) {
                null to e
            }
            val resp = result.first
            Api.ui {
                when {
                    result.second != null -> Ui.toast(this, R.string.backup_bad_code)
                    resp == null -> Unit
                    resp.ok -> { Ui.toast(this, R.string.backup_restored); recreate() }
                    resp.code == 401 -> Ui.toast(this, R.string.backup_need_login)
                    resp.code == 404 -> Ui.toast(this, R.string.backup_none)
                    resp.code == 0 -> Ui.toast(this, R.string.backup_offline)
                    else -> Ui.toast(this, getString(R.string.backup_failed, resp.code))
                }
            }
        }
    }

    /**
     * «Fjern inaktiv lagleder» (musingsUI runde 9): dialogen med jaktlag-valg
     * vises KUN når flere lag har inaktiv lagleder; ellers toast «Ingen inaktive
     * lagledere funnet». Inaktivitet krever aktivitetsdata per lagleder =
     * backend (backend_spec.md §11), så i skjelettet er lista tom.
     */
    private fun removeInactiveLeader() {
        val teams = teamsWithInactiveLeader()
        when {
            teams.isEmpty() -> Ui.toast(this, R.string.team_no_inactive_leaders)
            teams.size == 1 -> confirmRemoveInactive(teams[0])
            else -> AlertDialog.Builder(this)
                .setTitle(R.string.team_remove_inactive_which)
                .setItems(teams.map { it.name }.toTypedArray()) { _, i ->
                    confirmRemoveInactive(teams[i])
                }
                .setNegativeButton(R.string.cancel, null)
                .show()
        }
    }

    /** Backend (§11) leverer hvilke lag som har inaktiv lagleder; tom i skjelettet. */
    private fun teamsWithInactiveLeader(): List<Team> = emptyList()

    private fun confirmRemoveInactive(@Suppress("UNUSED_PARAMETER") t: Team) {
        // Backend: push til lagleder + 7-dagers nedtelling. Her kun kvittering.
        Ui.toast(this, R.string.team_backend_wait)
    }

    private fun devMenu() {
        AlertDialog.Builder(this)
            .setTitle(R.string.dev_menu)
            .setItems(arrayOf(getString(R.string.dev_generate),
                getString(R.string.dev_dummy_scan),
                getString(R.string.dev_always_startup) + ": " +
                    (if (store.alwaysShowStartup) "på" else "av"),
                getString(R.string.dev_add_friend),
                getString(R.string.dev_api_url),
                getString(R.string.dev_backup_selftest))) { _, which ->
                when (which) {
                    0 -> DevTools.generateSeries(this)
                    1 -> DevTools.dummyScan(this)
                    2 -> store.alwaysShowStartup = !store.alwaysShowStartup
                    3 -> DevTools.addFriendDialog(this)
                    4 -> apiUrlDialog()
                    // Verifiserer blobformatet uten server og uten innlogging.
                    5 -> Api.io {
                        val r = Backup.selfTest(this)
                        Api.ui { AlertDialog.Builder(this)
                            .setTitle(R.string.dev_backup_selftest)
                            .setMessage(r)
                            .setPositiveButton(R.string.ok, null).show() }
                    }
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    /**
     * Utvikler: pek appen mot en annen backend (lokal maskin, staging) uten å
     * bygge på nytt. Tom verdi = BuildConfig.API_BASE_URL.
     */
    private fun apiUrlDialog() {
        val field = android.widget.EditText(this).apply {
            setText(store.apiBaseUrl)
            hint = Api.baseUrl(this@AvansertActivity)
            inputType = android.text.InputType.TYPE_TEXT_VARIATION_URI
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.dev_api_url)
            .setView(Ui.col(this, 16).apply { addView(field) })
            .setPositiveButton(R.string.save) { _, _ ->
                store.apiBaseUrl = field.text.toString().trim()
                Ui.toast(this, Api.baseUrl(this))
            }
            .setNeutralButton(R.string.dev_api_url_reset) { _, _ ->
                store.apiBaseUrl = ""
                Ui.toast(this, Api.baseUrl(this))
            }
            .setNegativeButton(R.string.cancel, null)
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
