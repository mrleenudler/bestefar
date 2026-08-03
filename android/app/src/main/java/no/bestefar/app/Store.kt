package no.bestefar.app

import android.content.Context
import org.json.JSONArray
import java.io.File
import java.time.LocalDate
import java.time.Instant
import java.time.ZoneId
import java.util.UUID

/**
 * Lokal persistens (offline-først, spec §1): serier og jaktlogg som JSON-filer
 * i appens filkatalog, innstillinger i SharedPreferences. Ingen konto kreves;
 * opplasting køes (uploaded=false) til backend-kobling kommer.
 */
class Store private constructor(ctx: Context) {

    companion object {
        @Volatile private var instance: Store? = null
        fun get(context: Context): Store =
            instance ?: synchronized(this) {
                instance ?: Store(context.applicationContext).also { instance = it }
            }

        fun newId(): String = UUID.randomUUID().toString()

        /** Jaktår: 1. april–31. mars (spec §5). Nøkkel = startåret. */
        fun seasonKey(ts: Long): Int {
            val d = Instant.ofEpochMilli(ts).atZone(ZoneId.systemDefault()).toLocalDate()
            return if (d.monthValue >= 4) d.year else d.year - 1
        }
        fun seasonLabel(key: Int) = "$key/${(key + 1) % 100}"
    }

    private val prefs = ctx.getSharedPreferences("bestefar_ui", Context.MODE_PRIVATE)
    private val seriesFile = File(ctx.filesDir, "series.json")
    private val huntFile = File(ctx.filesDir, "hunts.json")

    private var seriesCache: MutableList<SeriesRecord>? = null
    private var huntCache: MutableList<HuntRecord>? = null

    // ---------- Våpenkartotek ----------

    fun weapons(): List<Weapon> {
        val raw = prefs.getString("weapons", null) ?: return emptyList()
        val arr = JSONArray(raw)
        return (0 until arr.length()).map { Weapon.fromJson(arr.getJSONObject(it)) }
    }

    fun saveWeapons(list: List<Weapon>) {
        val arr = JSONArray().apply { list.forEach { put(it.toJson()) } }
        prefs.edit().putString("weapons", arr.toString()).apply()
    }

    fun addWeapon(w: Weapon) = saveWeapons(weapons() + w)

    fun updateWeapon(w: Weapon) = saveWeapons(weapons().map { if (it.id == w.id) w else it })

    var selectedWeaponId: String?
        get() = prefs.getString("selectedWeaponId", null)
        set(v) { prefs.edit().putString("selectedWeaponId", v).apply() }

    // ---------- Optikkprofiler ----------

    fun optics(): List<OpticProfile> {
        val raw = prefs.getString("optics", null) ?: return emptyList()
        val arr = JSONArray(raw)
        return (0 until arr.length()).map { OpticProfile.fromJson(arr.getJSONObject(it)) }
    }

    fun saveOptics(list: List<OpticProfile>) {
        val arr = JSONArray().apply { list.forEach { put(it.toJson()) } }
        prefs.edit().putString("optics", arr.toString()).apply()
    }

    fun addOptic(o: OpticProfile) = saveOptics(optics() + o)

    fun updateOptic(o: OpticProfile) =
        saveOptics(optics().map { if (it.id == o.id) o else it })

    /** Effektiv klikkverdi for våpen: optikkprofil foran legacy-feltet. */
    fun clickCmFor(w: Weapon?): Double? {
        if (w == null) return null
        val optic = w.opticId?.let { id -> optics().firstOrNull { it.id == id } }
        return optic?.clickCmPer100 ?: w.clickValueCm
    }

    /** Egendefinert avstand på X-knappen (musingsUI); 0 = ikke satt. */
    var customDistance: Int
        get() = prefs.getInt("customDistance", 0)
        set(v) { prefs.edit().putInt("customDistance", v).apply() }

    /** Ett registrert våpen -> automatisk valgt uten prompt (spec §1/§2). */
    fun selectedWeapon(): Weapon? {
        val ws = weapons()
        if (ws.size == 1) return ws[0]
        return ws.firstOrNull { it.id == selectedWeaponId }
    }

    /** Dato (ISO) våpen sist ble bekreftet — prompt én gang per dag. */
    var weaponConfirmedDate: String
        get() = prefs.getString("weaponConfirmedDate", "") ?: ""
        set(v) { prefs.edit().putString("weaponConfirmedDate", v).apply() }

    fun weaponNeedsDayConfirm(): Boolean =
        weapons().size > 1 && weaponConfirmedDate != LocalDate.now().toString()

    // ---------- Øktkontekst ----------

    var distanceM: Int
        get() = prefs.getInt("distanceM", 100)
        set(v) { prefs.edit().putInt("distanceM", v).apply() }

    /** «Ikke spør — manuell stilling» (spec §2). */
    var manualPosition: Boolean
        get() = prefs.getBoolean("manualPosition", false)
        set(v) { prefs.edit().putBoolean("manualPosition", v).apply() }

    var currentPosition: Position
        get() = Position.of(prefs.getString("currentPosition", null))
        set(v) { prefs.edit().putString("currentPosition", v.name).apply() }

    var currentModifier: PosModifier
        get() = PosModifier.of(prefs.getString("currentModifier", null))
        set(v) { prefs.edit().putString("currentModifier", v.name).apply() }

    /** Siste modifikator huskes per hovedstilling (spec §2). */
    fun lastModifier(p: Position): PosModifier =
        PosModifier.of(prefs.getString("mod_${p.name}", null))

    fun setLastModifier(p: Position, m: PosModifier) =
        prefs.edit().putString("mod_${p.name}", m.name).apply()

    /** Øvelsesmodus: stilling forhåndsvalgt av øvelsesmotoren (spec §5). */
    var practicePosition: Position?
        get() = prefs.getString("practicePosition", null)?.let { Position.of(it) }
        set(v) { prefs.edit().putString("practicePosition", v?.name).apply() }

    // ---------- Skytterprofil ----------

    var birthYear: Int
        get() = prefs.getInt("birthYear", 0)
        set(v) { prefs.edit().putInt("birthYear", v).apply() }

    var skytterlag: String
        get() = prefs.getString("skytterlag", "") ?: ""
        set(v) { prefs.edit().putString("skytterlag", v).apply() }

    var jaktlag: String
        get() = prefs.getString("jaktlag", "") ?: ""
        set(v) { prefs.edit().putString("jaktlag", v).apply() }

    /** Akseptabel skadeskytingsrate — felles forsvarlighetskriterium (spec §1). */
    var rateLimit: Double
        get() = prefs.getFloat("rateLimit", 0.05f).toDouble()
        set(v) { prefs.edit().putFloat("rateLimit", v.toFloat()).apply() }

    // ---------- Samtykker ----------
    // Verdier: "" (aldri spurt) / "ja" / "senere" / "aldri"

    var consentResearch: String
        get() = prefs.getString("consentResearch", "") ?: ""
        set(v) { prefs.edit().putString("consentResearch", v).apply() }

    var consentHunt: String
        get() = prefs.getString("consentHunt", "") ?: ""
        set(v) { prefs.edit().putString("consentHunt", v).apply() }

    /** Serieantall ved forrige samtykkeprompt (5, deretter hver 10., spec §1). */
    var consentLastPromptCount: Int
        get() = prefs.getInt("consentLastPromptCount", 0)
        set(v) { prefs.edit().putInt("consentLastPromptCount", v).apply() }

    var shareGranularity: String
        get() = prefs.getString("shareGranularity", "Kommune (~10 km)") ?: ""
        set(v) { prefs.edit().putString("shareGranularity", v).apply() }

    /** Sesong (jaktår-startår) da forskningssamtykke sist ble bekreftet; brukes
     *  til å spørre på nytt hver nye sesong (musingsUI runde 8). -1 = aldri. */
    var researchConsentSeason: Int
        get() = prefs.getInt("researchConsentSeason", -1)
        set(v) { prefs.edit().putInt("researchConsentSeason", v).apply() }

    // ---------- Jaktmodus ----------

    var huntMode: Boolean
        get() = prefs.getBoolean("huntMode", false)
        set(v) { prefs.edit().putBoolean("huntMode", v).apply() }

    var locationLogging: Boolean
        get() = prefs.getBoolean("locationLogging", false)
        set(v) { prefs.edit().putBoolean("locationLogging", v).apply() }

    var huntSpecies: Set<String>
        get() = prefs.getStringSet("huntSpecies", emptySet()) ?: emptySet()
        set(v) { prefs.edit().putStringSet("huntSpecies", v).apply() }

    /** Tutorial-overlegg vist (erstatter intro-skjermene, musingsUI). */
    var tutorialSeen: Boolean
        get() = prefs.getBoolean("tutorialSeen", false)
        set(v) { prefs.edit().putBoolean("tutorialSeen", v).apply() }

    /**
     * Versjon av oppstartsmeldingen som er sett. Endres teksten (STARTUP_MSG_VERSION
     * bumpes), vises den på nytt (musingsUI runde 4).
     */
    var startupMsgSeenVersion: Int
        get() = prefs.getInt("startupMsgSeenVersion", 0)
        set(v) { prefs.edit().putInt("startupMsgSeenVersion", v).apply() }

    /** Tema: "system" | "light" | "dark". Default lys (musingsUI runde 5). */
    var themeMode: String
        get() = prefs.getString("themeMode", "light") ?: "light"
        set(v) { prefs.edit().putString("themeMode", v).apply(); BestefarApp.applyTheme(v) }

    /** Venstrehåndsmodus: speiler UI horisontalt (musingsUI runde 5). */
    var leftHanded: Boolean
        get() = prefs.getBoolean("leftHanded", false)
        set(v) { prefs.edit().putBoolean("leftHanded", v).apply() }

    /** Utvikler: vis oppstartsmelding hver gang (musingsUI runde 5). */
    var alwaysShowStartup: Boolean
        get() = prefs.getBoolean("alwaysShowStartup", false)
        set(v) { prefs.edit().putBoolean("alwaysShowStartup", v).apply() }

    /**
     * Vis «Vil du dele bilder …»-vinduet ved oppstart (musingsUI runde 7).
     * Kan slås av i Avanserte innstillinger. Default på.
     */
    var startupDonateAsk: Boolean
        get() = prefs.getBoolean("startupDonateAsk", true)
        set(v) { prefs.edit().putBoolean("startupDonateAsk", v).apply() }

    /** «Mitt jaktmål» tilbys etter tre serier (musingsUI runde 4). */
    var jaktmaalPromptSeen: Boolean
        get() = prefs.getBoolean("jaktmaalPromptSeen", false)
        set(v) { prefs.edit().putBoolean("jaktmaalPromptSeen", v).apply() }

    /** Serieantall da jaktmål ble vist; forskningsdialog tidligst 2 serier etter. */
    var jaktmaalPromptCount: Int
        get() = prefs.getInt("jaktmaalPromptCount", 0)
        set(v) { prefs.edit().putInt("jaktmaalPromptCount", v).apply() }

    var rateChosen: Boolean
        get() = prefs.getBoolean("rateChosen", false)
        set(v) { prefs.edit().putBoolean("rateChosen", v).apply() }

    var nickname: String
        get() = prefs.getString("nickname", "") ?: ""
        set(v) { prefs.edit().putString("nickname", v).apply() }

    /** «La venner finne meg», default av (musingsUI runde 3). */
    var findable: Boolean
        get() = prefs.getBoolean("findable", false)
        set(v) { prefs.edit().putBoolean("findable", v).apply() }

    /** Delingsvalg mot forskning (Vilt/Dato/Posisjon/Skuddsituasjon). */
    var researchShare: Set<String>
        get() = prefs.getStringSet("researchShare", emptySet()) ?: emptySet()
        set(v) { prefs.edit().putStringSet("researchShare", v).apply() }

    /** Delingsvalg mot venner. */
    var friendShare: Set<String>
        get() = prefs.getStringSet("friendShare", emptySet()) ?: emptySet()
        set(v) { prefs.edit().putStringSet("friendShare", v).apply() }

    var friendShareActive: Boolean
        get() = prefs.getBoolean("friendShareActive", false)
        set(v) { prefs.edit().putBoolean("friendShareActive", v).apply() }

    /**
     * Deling av «bilder der appen gjør en dårlig jobb» til utvikler
     * (oppstartsmelding vindu 2, musingsUI runde 4). "" = ikke spurt.
     */
    var shareDevImages: String
        get() = prefs.getString("shareDevImages", "") ?: ""
        set(v) { prefs.edit().putString("shareDevImages", v).apply() }
    val shareDevImagesActive: Boolean get() = shareDevImages == "ja"

    /** Sesong da bildedelings-spørsmålet sist ble stilt ved oppstart; brukes til
     *  å spørre én gang neste sesong hvis deling ikke er valgt (musingsUI r9). */
    var shareDevImagesSeason: Int
        get() = prefs.getInt("shareDevImagesSeason", -1)
        set(v) { prefs.edit().putInt("shareDevImagesSeason", v).apply() }

    /**
     * Lagre scannede skjermbilder i telefonens bildearkiv (musingsUI runde 10).
     * Default PÅ til spørsmålet er stilt etter første scan — bildet er det mest
     * verdifulle vi har hvis analysen bommer, og svarer brukeren «Nei» slettes
     * også det første bildet.
     */
    var saveScansToGallery: Boolean
        get() = prefs.getBoolean("saveScansToGallery", true)
        set(v) { prefs.edit().putBoolean("saveScansToGallery", v).apply() }

    var saveScansAsked: Boolean
        get() = prefs.getBoolean("saveScansAsked", false)
        set(v) { prefs.edit().putBoolean("saveScansAsked", v).apply() }

    // ---------- Opplasting (backend_spec §6) ----------

    /**
     * Kun wifi. Default PÅ: køen består av fullskala-JPEG-er, og en jeger på
     * skytebanen skal ikke bruke opp mobildata uten å ha bedt om det.
     */
    var uploadWifiOnly: Boolean
        get() = prefs.getBoolean("uploadWifiOnly", true)
        set(v) { prefs.edit().putBoolean("uploadWifiOnly", v).apply() }

    /** Tidspunkt for siste fullførte tømming av køen; 0 = aldri. */
    var lastSyncTs: Long
        get() = prefs.getLong("lastSyncTs", 0L)
        set(v) { prefs.edit().putLong("lastSyncTs", v).apply() }

    /** Utvikler: overstyr API-adressen i felt. Tom = BuildConfig.API_BASE_URL. */
    var apiBaseUrl: String
        get() = prefs.getString("apiBaseUrl", "") ?: ""
        set(v) { prefs.edit().putString("apiBaseUrl", v).apply() }

    // ---------- Serier ----------

    @Synchronized
    fun allSeries(): List<SeriesRecord> {
        seriesCache?.let { return it }
        val list = mutableListOf<SeriesRecord>()
        if (seriesFile.exists()) {
            try {
                val arr = JSONArray(seriesFile.readText())
                for (i in 0 until arr.length()) list.add(SeriesRecord.fromJson(arr.getJSONObject(i)))
            } catch (_: Exception) { /* korrupt fil -> start tomt, ikke krasj */ }
        }
        seriesCache = list
        return list
    }

    @Synchronized
    fun addSeries(r: SeriesRecord) {
        val list = allSeries().toMutableList().apply { add(r) }
        seriesCache = list
        writeSeries(list)
    }

    @Synchronized
    fun updateSeries(r: SeriesRecord) {
        val list = allSeries().map { if (it.id == r.id) r else it }.toMutableList()
        seriesCache = list
        writeSeries(list)
    }

    @Synchronized
    fun deleteSeries(ids: Collection<String>) {
        val list = allSeries().filter { it.id !in ids }.toMutableList()
        seriesCache = list
        writeSeries(list)
    }

    private fun writeSeries(list: List<SeriesRecord>) =
        seriesFile.writeText(JSONArray().apply { list.forEach { put(it.toJson()) } }.toString())

    fun currentSeasonSeries(): List<SeriesRecord> {
        val now = seasonKey(System.currentTimeMillis())
        return allSeries().filter { seasonKey(it.ts) == now }
    }

    fun seriesToday(): List<SeriesRecord> {
        val today = LocalDate.now()
        return allSeries().filter {
            Instant.ofEpochMilli(it.ts).atZone(ZoneId.systemDefault()).toLocalDate() == today
        }
    }

    /** Antall øvelsesskudd denne sesongen (startskjerm-teller, musingsUI r4). */
    fun shotsThisSeason(): Int = currentSeasonSeries().sumOf { it.shots.size }

    /** Antall serier skutt fra hver stilling denne sesongen (stillingsvalg). */
    fun seriesCountByPosition(): Map<Position, Int> =
        currentSeasonSeries().groupingBy { it.position }.eachCount()

    /** Antall SKUDD skutt fra hver stilling denne sesongen (stillingsvelger viser
     *  skudd, ikke serier — musingsUI runde 8). */
    fun shotsCountByPosition(): Map<Position, Int> =
        currentSeasonSeries().groupBy { it.position }
            .mapValues { e -> e.value.sumOf { it.shots.size } }

    /** Sesongens serier for ett våpen, kronologisk (innskytingssjekk). */
    fun seasonSeriesForWeapon(weaponId: String?): List<SeriesRecord> =
        currentSeasonSeries().filter { it.weaponId == weaponId }.sortedBy { it.ts }

    // ---------- Jaktlogg ----------

    @Synchronized
    fun allHunts(): List<HuntRecord> {
        huntCache?.let { return it }
        val list = mutableListOf<HuntRecord>()
        if (huntFile.exists()) {
            try {
                val arr = JSONArray(huntFile.readText())
                for (i in 0 until arr.length()) list.add(HuntRecord.fromJson(arr.getJSONObject(i)))
            } catch (_: Exception) { }
        }
        huntCache = list
        return list
    }

    @Synchronized
    fun addHunt(r: HuntRecord) {
        val list = allHunts().toMutableList().apply { add(r) }
        huntCache = list
        writeHunts(list)
    }

    @Synchronized
    fun updateHunt(r: HuntRecord) {
        val list = allHunts().map { if (it.id == r.id) r else it }.toMutableList()
        huntCache = list
        writeHunts(list)
    }

    @Synchronized
    fun deleteHunts(ids: Collection<String>) {
        val list = allHunts().filter { it.id !in ids }.toMutableList()
        huntCache = list
        writeHunts(list)
    }

    private fun writeHunts(list: List<HuntRecord>) =
        huntFile.writeText(JSONArray().apply { list.forEach { put(it.toJson()) } }.toString())

    // ---------- Venner og lag (FRONT-END-SKJELETT, jf. backend_spec.md) ----------

    fun friends(): List<Friend> {
        val raw = prefs.getString("friends", null) ?: return emptyList()
        val arr = JSONArray(raw)
        return (0 until arr.length()).map { Friend.fromJson(arr.getJSONObject(it)) }
    }

    fun saveFriends(list: List<Friend>) {
        prefs.edit().putString("friends",
            JSONArray().apply { list.forEach { put(it.toJson()) } }.toString()).apply()
    }

    fun addFriend(f: Friend) = saveFriends(friends() + f)
    fun updateFriend(f: Friend) = saveFriends(friends().map { if (it.id == f.id) f else it })

    fun teams(): List<Team> {
        val raw = prefs.getString("teams", null) ?: return emptyList()
        val arr = JSONArray(raw)
        return (0 until arr.length()).map { Team.fromJson(arr.getJSONObject(it)) }
    }

    fun saveTeams(list: List<Team>) {
        prefs.edit().putString("teams",
            JSONArray().apply { list.forEach { put(it.toJson()) } }.toString()).apply()
    }

    fun addTeam(t: Team) = saveTeams(teams() + t)

    /** Sletting: lokal wipe (spec §6). Sletteanmodning via pseudonym-ID kommer med backend. */
    @Synchronized
    fun wipeAll() {
        seriesFile.delete(); huntFile.delete()
        seriesCache = null; huntCache = null
        prefs.edit().clear().apply()
    }
}
