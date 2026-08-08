package no.bestefar.app

import android.content.Context
import android.graphics.Color
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** Små byggeklosser for programmatisk UI (holder fragmentene kompakte). */
object Ui {

    fun dp(c: Context, v: Int): Int = (v * c.resources.displayMetrics.density).toInt()

    fun col(c: Context, padDp: Int = 16): LinearLayout = LinearLayout(c).apply {
        orientation = LinearLayout.VERTICAL
        val p = dp(c, padDp)
        setPadding(p, p, p, p)
    }

    fun row(c: Context): LinearLayout = LinearLayout(c).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
    }

    fun scroll(c: Context, content: View): ScrollView = ScrollView(c).apply {
        isFillViewport = true
        addView(content, ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT)
    }

    fun title(c: Context, s: CharSequence): TextView = TextView(c).apply {
        text = s; textSize = 22f
        setPadding(0, 0, 0, dp(c, 8))
    }

    fun section(c: Context, s: CharSequence): TextView = TextView(c).apply {
        text = s; textSize = 17f
        setTextColor(themeColor(c, com.google.android.material.R.attr.colorPrimary))
        setPadding(0, dp(c, 20), 0, dp(c, 4))
    }

    fun body(c: Context, s: CharSequence): TextView = TextView(c).apply {
        text = s; textSize = 15f
        setPadding(0, dp(c, 4), 0, dp(c, 4))
    }

    fun hint(c: Context, s: CharSequence): TextView = TextView(c).apply {
        text = s; textSize = 13f; alpha = 0.65f
        setPadding(0, dp(c, 4), 0, dp(c, 4))
    }

    fun vspace(c: Context, h: Int): View = View(c).apply {
        layoutParams = LinearLayout.LayoutParams(1, dp(c, h))
    }

    private val tidFmt = java.time.format.DateTimeFormatter.ofPattern(
        "d. MMMM yyyy 'kl.' HH:mm", java.util.Locale.forLanguageTag("no"))

    /**
     * ISO-8601 fra serveren -> norsk dato og klokkeslett i telefonens sone.
     *
     * Serveren sender med offset, med `Z`, eller nakent, derfor tre forsoek.
     * Feiler alle, gir vi TOM STRENG - en raa ISO-streng i et vindu er stoey,
     * ikke informasjon, og kallstedene skal kunne utelate linja helt.
     */
    fun norskTid(iso: String): String {
        if (iso.isBlank()) return ""
        val sone = java.time.ZoneId.systemDefault()
        val tid = try {
            java.time.OffsetDateTime.parse(iso).atZoneSameInstant(sone).toLocalDateTime()
        } catch (_: Exception) {
            try {
                java.time.Instant.parse(iso).atZone(sone).toLocalDateTime()
            } catch (_: Exception) {
                try { java.time.LocalDateTime.parse(iso) } catch (_: Exception) { null }
            }
        }
        return tid?.format(tidFmt) ?: ""
    }

    /**
     * Løs opp en tema-attributt til en farge. VIKTIG (musingsUI runde 8):
     * enkelte attributter — særlig android.R.attr.textColorPrimary — peker til en
     * ColorStateList, ikke en direkte farge-int. Da er `tv.data` en RESSURS-ID
     * (tolket som ARGB blir det en tilfeldig, ofte usynlig farge). Det var
     * rotårsaken til at silhuetter/tekst tintet med tekstfargen var «usynlige» i
     * både lys og mørk visning gjennom flere runder. Her håndteres begge tilfeller:
     * direkte farge-int (colorPrimary) OG ressurs-referanse (textColorPrimary).
     */
    fun themeColor(c: Context, attr: Int): Int {
        val tv = TypedValue()
        if (!c.theme.resolveAttribute(attr, tv, true)) return Color.BLACK
        // Direkte farge-int (f.eks. colorPrimary)
        if (tv.type in TypedValue.TYPE_FIRST_COLOR_INT..TypedValue.TYPE_LAST_COLOR_INT)
            return tv.data
        // Referanse til color/ColorStateList (f.eks. android:textColorPrimary)
        if (tv.resourceId != 0) {
            return try {
                androidx.core.content.ContextCompat.getColor(c, tv.resourceId)
            } catch (_: Exception) {
                androidx.core.content.res.ResourcesCompat.getColorStateList(
                    c.resources, tv.resourceId, c.theme)?.defaultColor ?: Color.BLACK
            }
        }
        return Color.BLACK
    }

    /**
     * Edge-to-edge-fix (musingsUI): targetSdk 35 tegner under status- og
     * navigasjonslinjen; alle skjermbilder padder rota med systembar-innsets.
     */
    fun applyInsets(root: View) {
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(root) { v, wi ->
            val i = wi.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars())
            v.setPadding(i.left, i.top, i.right, i.bottom)
            wi
        }
    }

    /**
     * Svarte systemlinjer — TEGNET AV OSS (musingsUI runde 12).
     *
     * `android:statusBarColor` + `windowOptOutEdgeToEdgeEnforcement` i themes.xml
     * holdt bare fram til targetSdk 36: fra og med da ignorerer Android
     * opt-out-en, edge-to-edge tvinges, og systemlinjene tegnes rett oppå
     * appbakgrunnen. I MØRK visning merkes det ikke (bakgrunnen er nesten svart
     * fra før), men i LYS visning står de hvite systemikonene på lys brunt og
     * blir uleselige. Derfor legger vi selv en svart flate bak hver linje.
     *
     * Flatene legges i `android.R.id.content` ETTER appens rot, så de tegnes
     * over den; høyden settes fra innsettene, som er 0 på enheter der
     * opt-out-en fortsatt virker. Da blir dette en no-op i stedet for en
     * dobbel-tegning.
     */
    fun paintSystemBars(a: android.app.Activity) {
        val content = a.findViewById<android.widget.FrameLayout>(android.R.id.content) ?: return
        fun scrim(gravity: Int) = View(a).apply {
            setBackgroundColor(Color.BLACK)
            layoutParams = android.widget.FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, gravity)
        }
        val top = scrim(Gravity.TOP)
        val bottom = scrim(Gravity.BOTTOM)
        content.addView(top)
        content.addView(bottom)
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(content) { _, wi ->
            val i = wi.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars())
            top.layoutParams = top.layoutParams.also { it.height = i.top }
            bottom.layoutParams = bottom.layoutParams.also { it.height = i.bottom }
            wi
        }
        // Hvite systemikoner (svart bakgrunn) uansett lys/mørk visning.
        androidx.core.view.WindowCompat.getInsetsController(a.window, a.window.decorView)
            .apply {
                isAppearanceLightStatusBars = false
                isAppearanceLightNavigationBars = false
            }
    }

    /**
     * Dialogbygger for DESTRUKTIVE valg (musingsUI runde 12): rød
     * advarselstrekant i tittelen. Brukes der noe forsvinner og ikke kan hentes
     * tilbake — sletting av skudd, serier og «slett alle data». Bekreftelser
     * som bare er et veivalg (f.eks. «lik serie, lagre likevel?») skal IKKE ha
     * den; da slites ikonet ut og slutter å bety noe.
     */
    fun warningDialog(a: android.app.Activity): androidx.appcompat.app.AlertDialog.Builder =
        dangerDialog(a, R.drawable.ic_warning, R.string.warning_title)

    /**
     * Som [warningDialog], men med STOP-skilt (musingsUI runde 13). Forbeholdt
     * det ene valget som ikke bare sletter noe, men ALT — «Slett alle data».
     * Trekanten betyr «tenk deg om»; åttekanten betyr «her stopper du».
     * Skillet er verdiløst hvis det brukes to steder, så det brukes ett.
     */
    fun stopDialog(a: android.app.Activity): androidx.appcompat.app.AlertDialog.Builder =
        dangerDialog(a, R.drawable.ic_stop, R.string.stop_title)

    private fun dangerDialog(a: android.app.Activity, iconRes: Int, titleRes: Int):
        androidx.appcompat.app.AlertDialog.Builder {
        val icon = androidx.core.content.ContextCompat.getDrawable(a, iconRes)?.mutate()
        icon?.setTint(android.graphics.Color.parseColor("#C62828"))
        return androidx.appcompat.app.AlertDialog.Builder(a)
            .setIcon(icon)
            .setTitle(titleRes)
    }

    /**
     * Equalizer-ikonet som følger enhver henvisning til «Avanserte
     * innstillinger» (musingsUI runde 12). Det er alltid klikkbart og går rett
     * dit — en tekst som forteller hvor valget bor, men ikke tar deg dit, er
     * bare en beskjed om å lete selv.
     */
    fun advancedIcon(a: android.app.Activity, sizeDp: Int = 28): View =
        android.widget.ImageButton(a).apply {
            setImageResource(R.drawable.ic_settings_sliders)
            background = null
            scaleType = android.widget.ImageView.ScaleType.FIT_CENTER
            androidx.core.widget.ImageViewCompat.setImageTintList(this,
                android.content.res.ColorStateList.valueOf(
                    themeColor(a, com.google.android.material.R.attr.colorPrimary)))
            contentDescription = a.getString(R.string.advanced_open)
            layoutParams = LinearLayout.LayoutParams(dp(a, sizeDp), dp(a, sizeDp))
            setOnClickListener {
                a.startActivity(android.content.Intent(a, AvansertActivity::class.java))
            }
        }

    fun matchWrap(topDp: Int = 0, c: Context? = null): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            if (c != null) topMargin = dp(c, topDp)
        }

    /**
     * Valg-knapp (musingsUI runde 3): uvalgte = tydelig outlined, valgt =
     * invertert (fylt). `small` gir mindre, rektangulær variant
     * (hjelpemidler-knappene).
     */
    fun choiceButton(c: Context, label: CharSequence, selected: Boolean,
                     small: Boolean = false, onClick: () -> Unit):
        com.google.android.material.button.MaterialButton {
        val b = if (selected) {
            com.google.android.material.button.MaterialButton(c)
        } else {
            com.google.android.material.button.MaterialButton(c, null,
                com.google.android.material.R.attr.materialButtonOutlinedStyle)
        }
        b.text = label
        if (small) {
            b.textSize = 13f
            b.cornerRadius = dp(c, 6)
            b.minHeight = dp(c, 40)
            b.minimumHeight = dp(c, 40)
        }
        b.setOnClickListener { onClick() }
        return b
    }

    /** Fritekstfelt: stor forbokstav per default (musingsUI runde 3). */
    fun capitalize(e: android.widget.EditText) {
        e.inputType = android.text.InputType.TYPE_CLASS_TEXT or
            android.text.InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
    }

    /** Maks lengde på visningsnavn (musingsUI runde 4). */
    const val NAME_MAX = 24

    /**
     * Filter for visningsnavn (musingsUI runde 5): tillat vanlige latinske
     * tegn (inkl. æ ø å og aksenter), tall, mellomrom og enkel tegnsetting —
     * ikke bare ASCII. Begrens lengde.
     */
    fun nameFilters(): Array<android.text.InputFilter> = arrayOf(
        android.text.InputFilter.LengthFilter(NAME_MAX),
        android.text.InputFilter { src, s, e, _, _, _ ->
            val sb = StringBuilder()
            for (i in s until e) {
                val c = src[i]
                if ((c.isLetterOrDigit() && isLatin(c)) || c == ' ' || c in "-_.'") sb.append(c)
            }
            if (sb.length == e - s) null else sb   // null = uendret; ellers filtrert
        })

    private fun isLatin(c: Char): Boolean = when (Character.UnicodeBlock.of(c)) {
        Character.UnicodeBlock.BASIC_LATIN,
        Character.UnicodeBlock.LATIN_1_SUPPLEMENT,
        Character.UnicodeBlock.LATIN_EXTENDED_A,
        Character.UnicodeBlock.LATIN_EXTENDED_B -> true
        else -> false
    }

    /**
     * Global toast (musingsUI runde 5): ny toast avbryter forrige, så en kø av
     * toasts ikke blokkerer appen.
     */
    private var activeToast: android.widget.Toast? = null
    fun toast(c: Context, resId: Int) = toast(c, c.getString(resId))
    fun toast(c: Context, msg: String) {
        activeToast?.cancel()
        activeToast = android.widget.Toast.makeText(
            c.applicationContext, msg, android.widget.Toast.LENGTH_SHORT).also { it.show() }
    }

    /** Ikke tillat 0 som første siffer (musingsUI runde 5). */
    fun noLeadingZero(): android.text.InputFilter =
        android.text.InputFilter { source, s, e, _, dstart, _ ->
            if (dstart == 0 && e > s && source[s] == '0') "" else null
        }

    /**
     * Tillat en enslig «0», men ikke flere sifre etter (musingsUI runde 6):
     * «dyret løp» godtar 0 m, men «0» kan ikke bli «05».
     */
    fun singleZero(): android.text.InputFilter =
        android.text.InputFilter { _, _, _, dest, _, _ ->
            if (dest.toString() == "0") "" else null
        }

    /** Tekstboks med synlig ramme (kraftigere visuelt hint enn linje). */
    fun boxed(e: android.widget.EditText, c: Context) {
        e.background = android.graphics.drawable.GradientDrawable().apply {
            setColor(Color.TRANSPARENT)
            setStroke(dp(c, 1), Color.GRAY)
            cornerRadius = dp(c, 6).toFloat()
        }
        val p = dp(c, 8)
        e.setPadding(p, p, p, p)
    }
}
