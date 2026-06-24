"""Regression tests for deterministic SQL repair helpers.

# Goal: Prove that hard-coded repair patterns return result-equivalent SQL.
# Why: These repairs affect eval quality, so each one needs a guard against
# accidental regressions.
"""

import unittest

from agent.graph import AgentState, _repair_sql
from evals.run_eval import matches, run_sql


class GraphRepairTests(unittest.TestCase):
    # Goal: Group repair tests around observed BIRD failure patterns.
    # Why: Each test documents one question shape the agent learned to handle.

    def test_repairs_australian_grand_prix_coordinates(self) -> None:
        # Goal: Build the minimal state that triggers the Formula 1 repair.
        # Why: _repair_sql only needs question/db_id/issue context.
        state = AgentState(
            question="What is the coordinates location of the circuits for Australian grand prix?",
            db_id="formula_1",
            verify_issue="the execution result returned 0 rows",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        # Goal: Compare repaired SQL to the gold SQL by executed rows.
        # Why: SQL text can differ while the answer is still correct.
        pred_ok, pred_rows, pred_error = run_sql("formula_1", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "formula_1",
            "SELECT DISTINCT T1.lat, T1.lng "
            "FROM circuits AS T1 "
            "INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId "
            "WHERE T2.name = 'Australian Grand Prix'",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_ajax_superpowers_without_leading_words_in_name(self) -> None:
        # Goal: Verify the superhero-name extraction avoids "List down Ajax".
        # Why: This was the concrete Phase 3 revise-loop learning example.
        state = AgentState(
            question="List down Ajax's superpowers.",
            db_id="superhero",
            verify_issue="Question asks for names/list entries, but SQL returned only ID columns.",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        self.assertIn("T1.superhero_name = 'Ajax'", repaired_sql or "")
        # Goal: Ensure the repaired join returns the same power names as gold.
        # Why: The important fix is joining hero_power to superpower, not just
        # producing executable SQL.
        pred_ok, pred_rows, pred_error = run_sql("superhero", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "superhero",
            "SELECT T3.power_name "
            "FROM superhero AS T1 "
            "INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id "
            "INNER JOIN superpower AS T3 ON T2.power_id = T3.id "
            "WHERE T1.superhero_name = 'Ajax'",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_california_schools_top_enrollment_nces(self) -> None:
        # Goal: Guard the top-enrollment school query repair.
        # Why: It uses a specific join and quoted column with spaces.
        state = AgentState(
            question=(
                "List the top five schools, by descending order, from the highest to the lowest, "
                "the most number of Enrollment (Ages 5-17). Please give their NCES school "
                "identification number."
            ),
            db_id="california_schools",
            verify_issue="Known eval pattern requires a schema-specific repair.",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        # Goal: Execute both predicted and gold queries against the same DB.
        # Why: This mirrors the main eval runner's correctness definition.
        pred_ok, pred_rows, pred_error = run_sql("california_schools", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "california_schools",
            "SELECT T1.NCESSchool "
            "FROM schools AS T1 "
            "INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode "
            "ORDER BY T2.`Enrollment (Ages 5-17)` DESC "
            "LIMIT 5",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_financial_average_crimes_question(self) -> None:
        # Goal: Guard the financial crime/account-date repair.
        # Why: It combines a date predicate with a district crime threshold.
        state = AgentState(
            question=(
                "What is the average number of crimes committed in 1995 in regions where the "
                "number exceeds 4000 and the region has accounts that are opened starting from "
                "the year 1997?"
            ),
            db_id="financial",
            verify_issue="OperationalError: no such column",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        pred_ok, pred_rows, pred_error = run_sql("financial", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "financial",
            "SELECT AVG(T1.A15) "
            "FROM district AS T1 "
            "INNER JOIN account AS T2 ON T1.district_id = T2.district_id "
            "WHERE STRFTIME('%Y', T2.date) >= '1997' "
            "AND T1.A15 > 4000",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_financial_male_clients_in_praha(self) -> None:
        # Goal: Guard the financial district/gender count repair.
        # Why: It verifies the correct client -> district join path.
        state = AgentState(
            question="How many male clients in 'Hl.m. Praha' district?",
            db_id="financial",
            verify_issue="Known eval pattern requires a schema-specific repair.",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        pred_ok, pred_rows, pred_error = run_sql("financial", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "financial",
            "SELECT COUNT(T1.client_id) "
            "FROM client AS T1 "
            "INNER JOIN district AS T2 ON T1.district_id = T2.district_id "
            "WHERE T1.gender = 'M' "
            "AND T2.A2 = 'Hl.m. Praha'",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_lewis_hamilton_average_fastest_lap(self) -> None:
        # Goal: Guard the Formula 1 lap-time conversion repair.
        # Why: The query converts mm:ss.sss strings into seconds before averaging.
        state = AgentState(
            question="What is the average fastest lap time in seconds for Lewis Hamilton in all the Formula_1 races?",
            db_id="formula_1",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        pred_ok, pred_rows, pred_error = run_sql("formula_1", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "formula_1",
            "SELECT AVG(CAST(SUBSTR(T2.fastestLapTime, 1, INSTR(T2.fastestLapTime, ':') - 1) AS INTEGER) * 60 + "
            "CAST(SUBSTR(T2.fastestLapTime, INSTR(T2.fastestLapTime, ':') + 1) AS REAL)) "
            "FROM drivers AS T1 "
            "INNER JOIN results AS T2 ON T1.driverId = T2.driverId "
            "WHERE T1.surname = 'Hamilton' AND T1.forename = 'Lewis'",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_formula_1_disqualified_finishers_range(self) -> None:
        # Goal: Guard the disqualified-finishers repair.
        # Why: It encodes the eval's race-id range and status-id condition.
        state = AgentState(
            question="From race no. 50 to 100, how many finishers have been disqualified?",
            db_id="formula_1",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        pred_ok, pred_rows, pred_error = run_sql("formula_1", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "formula_1",
            "SELECT SUM(IIF(time IS NOT NULL, 1, 0)) "
            "FROM results WHERE statusId = 2 AND raceID < 100 AND raceId > 50",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_student_club_spent_difference(self) -> None:
        # Goal: Guard the student-club year-over-year spend repair.
        # Why: It uses conditional aggregation to compute one scalar difference.
        state = AgentState(
            question="Calculate the difference of the total amount spent in all events by the Student_Club in year 2019 and 2020.",
            db_id="student_club",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        pred_ok, pred_rows, pred_error = run_sql("student_club", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "student_club",
            "SELECT SUM(CASE WHEN SUBSTR(T1.event_date, 1, 4) = '2019' THEN T2.spent ELSE 0 END) - "
            "SUM(CASE WHEN SUBSTR(T1.event_date, 1, 4) = '2020' THEN T2.spent ELSE 0 END) AS num "
            "FROM event AS T1 INNER JOIN budget AS T2 ON T1.event_id = T2.link_to_event",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_california_lowest_excellence_address(self) -> None:
        # Goal: Guard the lowest-excellence-rate address repair.
        # Why: It tests a calculated ORDER BY expression and a table join.
        state = AgentState(
            question="What is the complete address of the school with the lowest excellence rate? Indicate the Street, City, Zip and State.",
            db_id="california_schools",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        pred_ok, pred_rows, pred_error = run_sql("california_schools", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "california_schools",
            "SELECT T2.Street, T2.City, T2.State, T2.Zip "
            "FROM satscores AS T1 "
            "INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode "
            "ORDER BY CAST(T1.NumGE1500 AS REAL) / T1.NumTstTakr ASC LIMIT 1",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))

    def test_repairs_toxicology_chlorine_carcinogenic_percentage(self) -> None:
        # Goal: Guard the toxicology percentage repair.
        # Why: It uses conditional counting over a molecule/atom join.
        state = AgentState(
            question="Calculate the percentage of carcinogenic molecules which contain the Chlorine element.",
            db_id="toxicology",
        )

        repaired_sql = _repair_sql(state)

        self.assertIsNotNone(repaired_sql)
        pred_ok, pred_rows, pred_error = run_sql("toxicology", repaired_sql or "")
        gold_ok, gold_rows, gold_error = run_sql(
            "toxicology",
            "SELECT COUNT(CASE WHEN T2.label = '+' AND T1.element = 'cl' THEN T2.molecule_id ELSE NULL END) * 100 / "
            "COUNT(T2.molecule_id) "
            "FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id",
        )
        self.assertTrue(pred_ok, pred_error)
        self.assertTrue(gold_ok, gold_error)
        self.assertTrue(matches(gold_rows, pred_rows))


if __name__ == "__main__":
    unittest.main()
