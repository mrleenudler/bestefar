package no.bestefar.app

import android.content.Intent
import android.graphics.Typeface
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.ViewGroup
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import java.time.LocalDate

/**
 * Min profil (musingsUI runde 5): visningsnavn (latinske tegn), fødselsår
 * (2–120), lag-liste, «Mitt jaktmål» (uthevet tall + (i)), tema-veksler øverst
 * til høyre, fortløpende lagring. Avanserte innstillinger er egen knapp.
 */
class ProfilActivity : AppCompatActivity() {

    private lateinit var store: Store
    private lateinit var content: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store.get(this)
        content = Ui.col(this)
        val scroller = Ui.scroll(this, content)
        Ui.applyInsets(scroller)
        setContentView(scroller)
        rebuild()
    }

    override fun onResume() { super.onResume(); rebuild() }

    private fun watcher(onText: (String) -> Unit) = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        override fun afterTextChanged(s: Editable?) = onText(s?.toString() ?: "")
    }

    /** Navnet som sist ble godtatt av serveren; hindrer at vi sender det samme igjen. */
    private var sendtNavn: String = ""
    private var nickFelt: EditText? = null
    private var nickHint: TextView? = null

    /**
     * Forlater brukeren skjermen uten å ta fokus ut av feltet først, rekker
     * ikke fokuslytteren å fyre. Da sendes navnet her i stedet.
     *
     * En avvisning kan ikke vises på en skjerm som er borte — men den blir ikke
     * usynlig av det: neste gang profilen åpnes, fylles feltet fra serverens
     * navn, så det avviste navnet står ikke igjen og ser lagret ut.
     */
    override fun onPause() {
        super.onPause()
        val f = nickFelt ?: return
        sendVisningsnavn(f, nickHint ?: return)
    }

    /**
     * Sender visningsnavnet og viser utfallet **med det samme**.
     *
     * Moderasjonen på serveren er synkron, så svaret er endelig når det kommer:
     * godkjent og lagret, eller avvist og ikke lagret. Det finnes ingen
     * «venter på moderasjon»-tilstand å vise — regelsettet svarer ja eller nei,
     * og den manuelle køen finnes ikke (ÅP-B8). Ved avvisning vises serverens
     * egen begrunnelse ordrett; den er skrevet for å leses av brukeren, og en
     * omskrevet versjon ville sagt mindre presist hva som må endres.
     *
     * Uten konto sendes ingenting. Navnet er da bare et lokalt kallenavn på
     * venne- og lagskjermene, som uansett ikke er koblet til server ennå.
     */
    private fun sendVisningsnavn(felt: EditText, hint: TextView) {
        if (!Auth.isLoggedIn(this)) return
        val navn = felt.text.toString().trim()
        if (navn.isEmpty() || navn == sendtNavn) return
        Login.lagreVisningsnavn(this, navn) { godkjent, tekst ->
            if (isFinishing) return@lagreVisningsnavn
            if (godkjent) {
                sendtNavn = tekst
                store.accountName = tekst
                store.nickname = tekst
                if (felt.text.toString() != tekst) felt.setText(tekst)
                hint.visibility = android.view.View.GONE
            } else {
                // Det avviste navnet ble IKKE lagret paa serveren, saa feltet
                // skal ikke staa igjen og se lagret ut. Vi setter tilbake det
                // som faktisk gjelder.
                felt.error = tekst.ifBlank { getString(R.string.profile_name_failed) }
                hint.text = tekst.ifBlank { getString(R.string.profile_name_failed) }
                hint.visibility = android.view.View.VISIBLE
                val gjeldende = store.accountName
                if (gjeldende.isNotEmpty()) felt.setText(gjeldende)
            }
        }
    }

    private fun themeLabel() = when (store.themeMode) {
        "dark" -> getString(R.string.theme_dark)
        "system" -> getString(R.string.theme_system)
        else -> getString(R.string.theme_light)
    }

    private fun rebuild() {
        content.removeAllViews()

        val header = Ui.row(this)
        header.addView(Ui.title(this, getString(R.string.profile_title)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        header.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = themeLabel()   // viser gjeldende modus (musingsUI runde 5)
            setOnClickListener { themeDialog() }
        })
        content.addView(header)

        // Visningsnavnet. Fram til v0.23 skrev dette feltet BARE til en lokal
        // verdi som aldri ble sendt noe sted, mens serveren hadde sitt eget
        // display_name brukeren ikke kunne endre. To navn, og feltet som het
        // «Visningsnavn» var det som ikke ble vist til noen.
        val nick = EditText(this).apply {
            hint = getString(R.string.profile_display_hint)
            filters = Ui.nameFilters()
        }
        Ui.capitalize(nick)
        // Innlogget er serverens navn fasit — det er det vennene ser.
        nick.setText(if (Auth.isLoggedIn(this) && store.accountName.isNotEmpty())
                         store.accountName else store.nickname)
        nick.addTextChangedListener(watcher { store.nickname = it.trim() })
        content.addView(nick)
        val nickHint = Ui.hint(this, "")
        nickHint.visibility = android.view.View.GONE
        content.addView(nickHint)
        // Sendes naar feltet forlates, ikke per tastetrykk: et PUT per bokstav
        // ville vaert stoey, og moderasjonssvaret skal komme mens brukeren
        // fortsatt staar paa skjermen — ikke ved neste appstart.
        nick.setOnFocusChangeListener { _, harFokus ->
            if (!harFokus) sendVisningsnavn(nick, nickHint)
        }
        nickFelt = nick
        this.nickHint = nickHint

        val birthRow = Ui.row(this)
        birthRow.addView(TextView(this).apply {
            text = getString(R.string.profile_birth_label); textSize = 16f
        })
        val birth = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            filters = arrayOf(android.text.InputFilter.LengthFilter(4), Ui.noLeadingZero())
            minWidth = Ui.dp(this@ProfilActivity, 90)
            setText(if (store.birthYear == 0) "" else store.birthYear.toString())
        }
        birth.addTextChangedListener(watcher { txt ->
            val y = txt.toIntOrNull() ?: return@watcher
            val now = LocalDate.now().year
            if (y in (now - 120)..(now - 2)) store.birthYear = y
        })
        birthRow.addView(birth)
        content.addView(birthRow)

        // Mine jaktlag og skytterlag — samme rekkefølge som i Venner
        // (sortOrder, musingsUI runde 7)
        content.addView(Ui.section(this, getString(R.string.profile_teams_title)))
        store.teams().sortedBy { it.sortOrder }.forEach { t ->
            content.addView(MaterialButton(this, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = t.name
                layoutParams = Ui.matchWrap(2, this@ProfilActivity)
                // Klikk åpner laget (musingsUI runde 6)
                setOnClickListener {
                    startActivity(Intent(this@ProfilActivity, TeamPageActivity::class.java)
                        .putExtra(TeamPageActivity.EXTRA_TEAM_ID, t.id))
                }
            })
        }
        // «+ legg til nytt lag» — outlined, ikke uthevet (musingsUI runde 7)
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.team_create_plus)
            layoutParams = Ui.matchWrap(4, this@ProfilActivity)
            setOnClickListener { startActivity(Intent(this@ProfilActivity, LagActivity::class.java)) }
        })

        content.addView(CheckBox(this).apply {
            text = getString(R.string.profile_findable)
            isChecked = store.findable
            setOnCheckedChangeListener { _, on -> store.findable = on }
        })

        // Mitt jaktmål: (i) høyrejustert på overskriftslinjen (musingsUI runde 5)
        val goalHeader = Ui.row(this)
        goalHeader.addView(Ui.section(this, getString(R.string.jaktmaal_title)).apply {
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        goalHeader.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_info)
            background = null
            contentDescription = getString(R.string.jaktmaal_why)
            setOnClickListener {
                AlertDialog.Builder(this@ProfilActivity)
                    .setMessage(R.string.jaktmaal_info)
                    .setPositiveButton(R.string.ok, null).show()
            }
        })
        content.addView(goalHeader)

        // Setningen flyter naturlig over to linjer, «Endre» til slutt t.h.
        // (musingsUI runde 6). Måltallet er 2 pt større og bold.
        val goalRow = Ui.row(this)
        val label = Dialogs.rateLabel(store.rateLimit)
        val sentence = "${getString(R.string.jaktmaal_prefix)} $label " +
            getString(R.string.jaktmaal_suffix)
        val sp = android.text.SpannableString(sentence)
        val s = sentence.indexOf(label)
        if (s >= 0) {
            sp.setSpan(Typeface.BOLD.let { android.text.style.StyleSpan(it) },
                s, s + label.length, 0)
            sp.setSpan(android.text.style.RelativeSizeSpan(1.15f),
                s, s + label.length, 0)
        }
        goalRow.addView(TextView(this).apply {
            text = sp; textSize = 16f
            layoutParams = LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        goalRow.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.borderlessButtonStyle).apply {
            text = getString(R.string.change)
            setOnClickListener { Dialogs.jaktmaalDialog(this@ProfilActivity, store) { rebuild() } }
        })
        content.addView(goalRow)

        // Konto (backend_spec §1, v0.17). Knappen viser tilstanden i selve
        // teksten — «Logg inn» kontra «Konto: <navn>» — så brukeren slipper å
        // åpne siden for å finne ut om de er innlogget.
        content.addView(Ui.section(this, getString(R.string.login_section)))
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = if (Auth.isLoggedIn(this@ProfilActivity))
                getString(R.string.login_account_is,
                    store.accountName.ifEmpty { getString(R.string.login_unnamed) })
            else getString(R.string.login_title)
            layoutParams = Ui.matchWrap(4, this@ProfilActivity)
            setOnClickListener {
                startActivity(Intent(this@ProfilActivity, LoggInnActivity::class.java))
            }
        })
        content.addView(Ui.hint(this, getString(
            if (Auth.isLoggedIn(this)) R.string.login_hint_in
            else R.string.login_hint_out)))

        // Avanserte innstillinger som egen knapp -> undermeny. Equalizer-ikonet
        // (musingsUI runde 12) følger enhver henvisning til siden, så brukeren
        // lærer symbolet og finner tilbake.
        content.addView(MaterialButton(this, null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            text = getString(R.string.profile_advanced)
            icon = androidx.core.content.ContextCompat.getDrawable(
                this@ProfilActivity, R.drawable.ic_settings_sliders)
            iconGravity = MaterialButton.ICON_GRAVITY_TEXT_START
            layoutParams = Ui.matchWrap(20, this@ProfilActivity)
            setOnClickListener { startActivity(Intent(this@ProfilActivity, AvansertActivity::class.java)) }
        })
    }

    private fun themeDialog() {
        val modes = listOf("light" to getString(R.string.theme_light),
            "dark" to getString(R.string.theme_dark),
            "system" to getString(R.string.theme_system))
        AlertDialog.Builder(this)
            .setTitle(R.string.theme_choose)   // «Velg visningsprofil»
            .setItems(modes.map { it.second }.toTypedArray()) { _, i ->
                store.themeMode = modes[i].first
                recreate()
            }
            .show()
    }
}
