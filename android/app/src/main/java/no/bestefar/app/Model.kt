package no.bestefar.app

import org.json.JSONArray
import org.json.JSONObject

/**
 * Datamodell for UI-laget (spec v0.4). Bokmål-etiketter direkte i enumene er
 * bevisst for v1 (spec §10.3); flyttes til ressurser ved i18n.
 */

enum class Position(val label: String) {
    LIGGENDE("Liggende"),
    SITTENDE("Sittende"),
    KNESTAENDE("Knestående"),
    STAAENDE("Stående"),
    BENK("Benk");   // egen inngang ved innskyting; teller ikke i kompetanse

    companion object {
        val hoved = listOf(LIGGENDE, SITTENDE, KNESTAENDE, STAAENDE)
        fun of(n: String?) = entries.firstOrNull { it.name == n } ?: LIGGENDE
    }
}

enum class PosModifier(val label: String) {
    UTEN("Uten"), ANLEGG("Anlegg"), REIM("Reim");
    companion object { fun of(n: String?) = entries.firstOrNull { it.name == n } ?: UTEN }
}

enum class Species(val label: String) {
    ELG("Elg"), HJORT("Hjort"), VILLREIN("Villrein"),
    RAADYR("Rådyr"), VILLSVIN("Villsvin"),
    ANNET("Annet");   // holdes utenfor analyser (spec §4)
    companion object { fun of(n: String?) = entries.firstOrNull { it.name == n } ?: ELG }
}

/**
 * Vinkeltaksonomi: PROVISORISK 30°/60° + side/front/bak — IKKE verifisert,
 * venter på BH-primærkilde (spec §10.2). `soneFaktor` = krymping av minste
 * halvakse på dødelig sone — PLASSHOLDER-verdier.
 */
enum class Angle(val label: String, val soneFaktor: Double) {
    SIDE("Bredside", 1.0),
    SKRAA30("Skrå 30°", 0.85),
    SKRAA60("Skrå 60°", 0.55),
    FRONT("Forfra", 0.40),
    BAK("Bakfra", 0.35);
    companion object { fun of(n: String?) = entries.firstOrNull { it.name == n } ?: SIDE }
}

enum class Outcome(val label: String) {
    DOEDELIG("Dødelig"), SKADE("Skade"), BOM("Bom");
    companion object { fun of(n: String?) = entries.firstOrNull { it.name == n } ?: BOM }
}

enum class FollowUp(val label: String) {
    FELT("Felt/avlivet"), FUNNET_DOED("Funnet død"),
    FRISKMELDT("Friskmeldt"), IKKE_GJENFUNNET("Ikke gjenfunnet");
    companion object { fun of(n: String?) = entries.firstOrNull { it.name == n } }
}

data class Weapon(
    val id: String,
    var name: String,            // våpennavn (modell), f.eks. «Sauer 100 6,5×55»
    var clickValueCm: Double?,   // legacy: cm/klikk @100m; erstattes av optikkprofil
    var ammoSplit: Boolean,
    var ammoName: String,        // aktiv ammo (metadata når split er av)
    var displayName: String = "",   // brukerens eget visningsnavn
    var icon: String = "rifle",     // nøkkel i WeaponIcons
    var opticId: String? = null,    // kobling til OpticProfile
) {
    val shownName: String get() = displayName.ifBlank { name }

    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id); put("name", name)
        put("click", clickValueCm ?: JSONObject.NULL)
        put("ammoSplit", ammoSplit); put("ammo", ammoName)
        put("displayName", displayName); put("icon", icon)
        put("opticId", opticId ?: JSONObject.NULL)
    }
    companion object {
        fun fromJson(o: JSONObject) = Weapon(
            id = o.getString("id"),
            name = o.getString("name"),
            clickValueCm = if (o.isNull("click")) null else o.getDouble("click"),
            ammoSplit = o.optBoolean("ammoSplit", false),
            ammoName = o.optString("ammo", ""),
            displayName = o.optString("displayName", ""),
            icon = o.optString("icon", "rifle"),
            opticId = if (o.isNull("opticId")) null else o.optString("opticId"),
        )
    }
}

/** Ikonvalg for våpen (flere kommer; nøkkel persisteres i Weapon.icon). */
object WeaponIcons {
    val options = listOf(
        "rifle" to R.drawable.ic_menu_rifle,
        "scoped" to R.drawable.ic_weapon_scoped,
        "simple" to R.drawable.ic_tab_vapen,
        "target" to R.drawable.ic_weapon_target,
    )
    fun res(key: String): Int =
        options.firstOrNull { it.first == key }?.second ?: R.drawable.ic_menu_rifle
}

/**
 * Optikkprofil (musingsUI): navngitt profil som settes opp én gang og velges
 * fra. Én aktiv representasjon av klikkverdien (MOA / MRAD / cm@100m);
 * konverteringer til de inaktive vises ikke. SMOA er avansert flagg.
 */
data class OpticProfile(
    val id: String,
    var displayName: String,
    var brandModel: String,
    var repr: String,        // "MOA" | "MRAD" | "CM"
    var moaValue: Double,    // 1/8, 1/4, 1/2, 1
    var mradValue: Double,   // 0.1 eller 0.05
    var cmValue: Double,     // cm/klikk @100m (kun redigerbar under CM)
    var smoa: Boolean,       // tårnet bruker forenklet 1 tomme/100 yards
) {
    /** Effektiv klikkverdi i cm/100m — én kilde til sannhet av gangen. */
    val clickCmPer100: Double
        get() = when (repr) {
            "MOA" -> moaValue * (if (smoa) SMOA_CM_PER_100M else MOA_CM_PER_100M)
            "MRAD" -> mradValue * 10.0
            else -> cmValue
        }

    val reprLabel: String
        get() = when (repr) {
            "MOA" -> "${moaLabel(moaValue)} MOA" + if (smoa) " (SMOA)" else ""
            "MRAD" -> "${"%.2f".format(mradValue).trimEnd('0').trimEnd(',', '.')} mrad"
            else -> "%.2f cm/100m".format(cmValue)
        }

    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id); put("displayName", displayName); put("brandModel", brandModel)
        put("repr", repr); put("moa", moaValue); put("mrad", mradValue)
        put("cm", cmValue); put("smoa", smoa)
    }
    companion object {
        /** Ekte MOA: 1,047 tommer/100 yards = 2,908 cm/100 m. */
        const val MOA_CM_PER_100M = 2.908
        /** SMOA: 1,000 tomme/100 yards = 2,778 cm/100 m (~4,7 % avvik). */
        const val SMOA_CM_PER_100M = 2.778

        val MOA_STEPS = listOf(0.125, 0.25, 0.5, 1.0)
        val MRAD_STEPS = listOf(0.1, 0.05)

        fun moaLabel(v: Double): String = when (v) {
            0.125 -> "1/8"; 0.25 -> "1/4"; 0.5 -> "1/2"; 1.0 -> "1"
            else -> "%.3f".format(v)
        }

        fun fromJson(o: JSONObject) = OpticProfile(
            id = o.getString("id"),
            displayName = o.optString("displayName", ""),
            brandModel = o.optString("brandModel", ""),
            repr = o.optString("repr", "CM"),
            moaValue = o.optDouble("moa", 0.25),
            mradValue = o.optDouble("mrad", 0.1),
            cmValue = o.optDouble("cm", 1.0),
            smoa = o.optBoolean("smoa", false),
        )
    }
}

data class Shot(val decimal: Double, val integer: Int, val rRel: Double, val theta: Double) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("d", decimal); put("i", integer); put("r", rRel); put("t", theta)
    }
    companion object {
        fun fromJson(o: JSONObject) =
            Shot(o.getDouble("d"), o.getInt("i"), o.getDouble("r"), o.getDouble("t"))
    }
}

data class SeriesRecord(
    val id: String,
    val ts: Long,
    val weaponId: String?,
    val ammoName: String,
    val distanceM: Int,
    val position: Position,
    val modifier: PosModifier,
    var shots: List<Shot>,
    var corrected: Boolean = false,
    var sendToFailChannel: Boolean = false,
    var uploaded: Boolean = false,
) {
    val sumDecimal get() = shots.sumOf { it.decimal }
    val sumInteger get() = shots.sumOf { it.integer }
    /** Benk teller ikke i evidensgrunnlaget (spec §2, §8). */
    val countsInEvidence get() = position != Position.BENK

    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id); put("ts", ts)
        put("weaponId", weaponId ?: JSONObject.NULL); put("ammo", ammoName)
        put("distanceM", distanceM)
        put("position", position.name); put("modifier", modifier.name)
        put("shots", JSONArray().apply { shots.forEach { put(it.toJson()) } })
        put("corrected", corrected); put("sendFail", sendToFailChannel)
        put("uploaded", uploaded)
    }
    companion object {
        fun fromJson(o: JSONObject): SeriesRecord {
            val sh = o.getJSONArray("shots")
            return SeriesRecord(
                id = o.getString("id"), ts = o.getLong("ts"),
                weaponId = if (o.isNull("weaponId")) null else o.getString("weaponId"),
                ammoName = o.optString("ammo", ""),
                distanceM = o.getInt("distanceM"),
                position = Position.of(o.getString("position")),
                modifier = PosModifier.of(o.getString("modifier")),
                shots = (0 until sh.length()).map { Shot.fromJson(sh.getJSONObject(it)) },
                corrected = o.optBoolean("corrected", false),
                sendToFailChannel = o.optBoolean("sendFail", false),
                uploaded = o.optBoolean("uploaded", false),
            )
        }
    }
}

data class HuntRecord(
    val id: String,
    val ts: Long,
    val species: Species,
    val distanceM: Int,
    val angle: Angle,
    val moving: Boolean,
    var outcome: Outcome,
    var followUp: FollowUp? = null,
    var followUpAsked: Boolean = false,
    val lat: Double? = null,
    val lon: Double? = null,
    val weaponId: String? = null,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id); put("ts", ts); put("species", species.name)
        put("distanceM", distanceM); put("angle", angle.name); put("moving", moving)
        put("outcome", outcome.name)
        put("followUp", followUp?.name ?: JSONObject.NULL)
        put("followUpAsked", followUpAsked)
        put("lat", lat ?: JSONObject.NULL); put("lon", lon ?: JSONObject.NULL)
        put("weaponId", weaponId ?: JSONObject.NULL)
    }
    companion object {
        fun fromJson(o: JSONObject) = HuntRecord(
            id = o.getString("id"), ts = o.getLong("ts"),
            species = Species.of(o.getString("species")),
            distanceM = o.getInt("distanceM"),
            angle = Angle.of(o.getString("angle")),
            moving = o.optBoolean("moving", false),
            outcome = Outcome.of(o.getString("outcome")),
            followUp = if (o.isNull("followUp")) null else FollowUp.of(o.getString("followUp")),
            followUpAsked = o.optBoolean("followUpAsked", false),
            lat = if (o.isNull("lat")) null else o.getDouble("lat"),
            lon = if (o.isNull("lon")) null else o.getDouble("lon"),
            weaponId = if (o.isNull("weaponId")) null else o.getString("weaponId"),
        )
    }
}
