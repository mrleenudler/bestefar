package no.bestefar.app

import android.app.Activity
import android.content.Intent
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin
import kotlin.random.Random

/**
 * Utviklerverktøy (musingsUI runde 5). Slå av ved å sette ENABLED=false —
 * da forsvinner Utvikler-menyen fra Avanserte innstillinger.
 */
object DevTools {

    const val ENABLED = true

    /**
     * Semi-tilfeldig serie vektet mot midten, men også mot blinkens areal, så
     * flertallet av «skuddene» legger seg rundt 6–7–8. Lagres som en normal serie.
     */
    fun generateSeries(activity: Activity, shots: Int = 10) {
        val store = Store.get(activity)
        val list = (1..shots).map {
            // Normal(7, 1.6) i poeng gir tyngde rundt 6-8; areal-vekting ivaretas
            // ved at ytre ringer (lavere poeng) er lettere å treffe -> venstre hale.
            var dec = 7.0 + gaussian() * 1.6
            dec = dec.coerceIn(1.0, 10.8)
            val rRel = (10.9 - dec).coerceAtLeast(0.0)   // grov invers av poeng
            val theta = Random.nextDouble(0.0, 2 * Math.PI)
            Shot(decimal = (dec * 10).toInt() / 10.0,
                integer = floor(dec).toInt().coerceIn(0, 10),
                rRel = rRel, theta = theta)
        }
        store.addSeries(SeriesRecord(
            id = Store.newId(), ts = System.currentTimeMillis(),
            weaponId = store.selectedWeapon()?.id, ammoName = "",
            distanceM = store.distanceM,
            position = store.currentPosition, modifier = store.currentModifier,
            shots = list))
        Ui.toast(activity, "Generert serie: sum %.1f".format(list.sumOf { it.decimal }))
    }

    private fun gaussian(): Double {
        // Box–Muller
        val u1 = Random.nextDouble().coerceAtLeast(1e-9)
        val u2 = Random.nextDouble()
        return kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * cos(2 * Math.PI * u2)
    }

    /**
     * «Dummy scan»: fabrikkerer et resultat og åpner ResultActivity, så hele
     * resultat-/stilling-/lagringsflyten kan testes uten kamera. (Bruker ikke
     * Real 1.jpg direkte siden bildet ikke er pakket i appen; fabrikkert
     * treffsett er en enklere og mer stabil test.)
     */
    /**
     * «Legg til venn» (musingsUI runde 7): fabrikkerer en venn i et valgt lag
     * med 5 genererte øvelsesserier (50 skudd) for å teste venne-/lag-UI-et.
     * Krever minst ett lag (venner legges i et lag).
     */
    fun addFriendDialog(activity: Activity) {
        val store = Store.get(activity)
        val teams = store.teams()
        if (teams.isEmpty()) { Ui.toast(activity, R.string.dev_friend_no_team); return }
        val input = android.widget.EditText(activity).apply {
            hint = activity.getString(R.string.dev_friend_name_hint)
        }
        Ui.capitalize(input)
        androidx.appcompat.app.AlertDialog.Builder(activity)
            .setTitle(R.string.dev_add_friend)
            .setView(input)
            .setPositiveButton(activity.getString(R.string.save)) { _, _ ->
                val name = input.text.toString().trim()
                if (name.isEmpty()) return@setPositiveButton
                if (teams.size == 1) makeFriend(activity, name, teams[0].id)
                else androidx.appcompat.app.AlertDialog.Builder(activity)
                    .setTitle(R.string.dev_add_friend)
                    .setItems(teams.map { it.name }.toTypedArray()) { _, i ->
                        makeFriend(activity, name, teams[i].id)
                    }
                    .setNegativeButton(R.string.cancel, null)
                    .show()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun makeFriend(activity: Activity, name: String, teamId: String) {
        val store = Store.get(activity)
        // 5 serier á 10 skudd = 50 øvelsesskudd (musingsUI runde 7)
        val total = 50
        store.addFriend(Friend(
            id = Store.newId(), displayName = name, teamIds = listOf(teamId),
            phone = "9%07d".format(Random.nextInt(0, 10_000_000)),
            homeKommune = listOf("Trysil", "Rendalen", "Stor-Elvdal", "Åmot").random(),
            shotsTotal = total, shotsSeason = total))
        Ui.toast(activity, R.string.dev_friend_added)
    }

    fun dummyScan(activity: Activity) {
        val n = 10   // 10 skudd, som en normal serie (musingsUI runde 7)
        val decimals = DoubleArray(n)
        val integers = IntArray(n)
        val rrel = DoubleArray(n)
        val theta = DoubleArray(n)
        for (i in 0 until n) {
            val dec = (6.0 + gaussian() * 1.4).coerceIn(1.0, 10.8)
            decimals[i] = (dec * 10).toInt() / 10.0
            integers[i] = floor(dec).toInt().coerceIn(0, 10)
            rrel[i] = (10.9 - dec).coerceAtLeast(0.0)
            theta[i] = Random.nextDouble(0.0, 2 * Math.PI)
        }
        activity.startActivity(Intent(activity, ResultActivity::class.java)
            .putExtra(ResultActivity.EXTRA_STATUS, BestefarCore.OK)
            .putExtra(ResultActivity.EXTRA_N_HITS, n)
            .putExtra(ResultActivity.EXTRA_CONFIDENCE, 0.9)
            .putExtra(ResultActivity.EXTRA_DECIMALS, decimals)
            .putExtra(ResultActivity.EXTRA_INTEGERS, integers)
            .putExtra(ResultActivity.EXTRA_RREL, rrel)
            .putExtra(ResultActivity.EXTRA_THETA, theta))
    }
}
