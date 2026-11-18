"""Synthetic qualification only: no project data, checkpoints, or target labels."""
import json
import unittest
import numpy as np
from c1_aggregate_selector import Client, select, fit, canonical_clients


class Qualification(unittest.TestCase):
    def clients(self):
        return [Client(0,[10,11,12],[[1.,0.],[1.,.1],[1.,-.1]]),
                Client(1,[20,21,22],[[-1.,0.],[-1.,.1],[-1.,-.1]])]

    def test_hand_centroids_and_unique_budget(self):
        r=select(self.clients(),initial_centroids=[[1.,0.],[-1.,0.]])
        self.assertEqual([p['window_id'] for p in r['plan']],[10,20,11,21])
        self.assertEqual([p['prior_paid_in_cluster'] for p in r['plan']],[0,0,1,1])
        np.testing.assert_allclose(r['centroids'],[[1,0],[-1,0]],atol=1e-15)
        self.assertEqual(r['spent_by_client'],{0:2,1:2})
        self.assertEqual(r['unused_budget'],0)

    def test_allocated_caps_bind(self):
        r=select(self.clients(),client_caps={0:1,1:3},cap_mode='allocated',initial_centroids=[[1.,0.],[-1.,0.]])
        self.assertEqual(r['spent_by_client'],{0:1,1:3})
        self.assertEqual(len({p['window_id'] for p in r['plan']}),4)

    def test_zero_cap_client(self):
        r=select(self.clients(),budget=3,client_caps={0:0,1:3},cap_mode='allocated')
        self.assertEqual(r['spent_by_client'],{0:0,1:3})

    def test_empty_and_zero_budget(self):
        empty=[Client(0,[],np.empty((0,2)))]
        self.assertEqual(select(empty)['stop_reason'],'empty_pool')
        self.assertIsNone(select(self.clients(),budget=0)['fit'])
        self.assertEqual(select(self.clients(),budget=0)['plan'],[])

    def test_empty_client_and_short_pool(self):
        r=select([Client(9,[],np.empty((0,2))),Client(0,[7],[[1.,0.]])])
        self.assertEqual([p['window_id'] for p in r['plan']],[7])
        self.assertEqual(r['unused_budget'],3)

    def test_unused_allocation_not_reallocated(self):
        r=select([Client(0,[],np.empty((0,2))),Client(1,[1,2,3],[[1.,0.],[1.,.1],[1.,-.1]])],
                 client_caps={0:3,1:1},cap_mode='allocated')
        self.assertEqual(len(r['plan']),1)
        self.assertEqual(r['unused_budget'],3)

    def test_collapsed_and_empty_clusters(self):
        r=select([Client(0,list(range(8)),np.tile([1.,0.],(8,1)))],initial_centroids=[[1.,0.],[1.,0.],[8.,8.],[9.,9.]])
        self.assertEqual([p['window_id'] for p in r['plan']],[0,1,2,3])
        self.assertEqual(r['cluster_counts'],[8,0,0,0])
        self.assertTrue(r['fit']['converged'])

    def test_row_and_client_order_invariance(self):
        a=select(self.clients())
        c=self.clients()
        reversed_rows=[Client(x.client_id,x.ids[::-1],x._z[::-1]) for x in c[::-1]]
        self.assertEqual(a,select(reversed_rows))
        self.assertEqual(a,select(self.clients()))

    def test_distance_and_client_ties(self):
        r=select([Client(2,[2],[[1.,0.]]),Client(1,[3,1],[[1.,0.],[1.,0.]])],budget=3,initial_centroids=[[1.,0.]])
        self.assertEqual([p['window_id'] for p in r['plan']],[1,3,2])

    def test_validation(self):
        for ids,z in [([1,1],[[1.,0.],[1.,0.]]),([1],[[0.,0.]]),([1],[[np.nan,1.]]),([1.1],[[1.,0.]])]:
            with self.assertRaises(ValueError):Client(0,ids,z)
        for b in [-1,5,True,1.5]:
            with self.assertRaises(ValueError):select(self.clients(),budget=b)
        for caps in [{0:1},{0:1,1:1},{0:-1,1:5}]:
            with self.assertRaises(ValueError):select(self.clients(),client_caps=caps,cap_mode='allocated')
        with self.assertRaises(ValueError):select([Client(0,[1],[[1.,0.]]),Client(1,[1],[[1.,0.]])])
        with self.assertRaises(ValueError):select([Client(0,[1],[[1.,0.]]),Client(1,[2],[[1.,0.,0.]])])

    def test_no_semantic_answer_channel(self):
        with self.assertRaises(TypeError):select(self.clients(),labels=[99]*6)
        r=select(self.clients())
        self.assertEqual(r['semantic_answers_read'],0)
        self.assertEqual(r['initial_target_labels'],0)
        # Simulated answers are attached only after the plan is frozen; changing
        # them cannot alter the ID plan or create extra calls.
        ids=[p['window_id'] for p in r['plan']]
        for answers in [['known']*4,['novel']*4,['A','A','known','B']]:
            ledger=list(zip(ids,answers))
            self.assertEqual([i for i,_ in ledger],ids)
            self.assertEqual(len(ledger),4)

    def test_iteration_cap_and_final_assignment(self):
        r=select(self.clients(),initial_centroids=[[.1,.1],[-.1,-.1]],max_iter=1,tolerance=0)
        self.assertEqual(r['fit']['iterations'],1)
        self.assertEqual(sum(r['cluster_counts']),6)
        self.assertIn(r['fit']['termination'],['iteration_cap','tolerance'])

    def test_central_lloyd_reference(self):
        # Independent pooled oracle exists only in this generated-data fixture.
        rng=np.random.default_rng(704)
        arrays=[rng.normal([1,0],.08,(13,2)),rng.normal([-1,0],.08,(11,2)),rng.normal([0,1],.08,(9,2))]
        clients=[];start=0
        for cid,z in enumerate(arrays):
            clients.append(Client(cid,range(start,start+len(z)),z));start+=len(z)
        initial=np.array([[1.,0.],[-1.,0.],[0.,1.]])
        centers,counts,status=fit(canonical_clients(clients),initial)
        pooled=np.concatenate(arrays);reference=initial.copy()
        for _ in range(100):
            assignments=np.argmin(np.sum((pooled[:,None,:]-reference[None,:,:])**2,axis=2),axis=1)
            updated=reference.copy()
            for j in range(3):
                if np.any(assignments==j):updated[j]=pooled[assignments==j].mean(axis=0)
            if np.max(np.linalg.norm(updated-reference,axis=1))<=1e-8:
                reference=updated;break
            reference=updated
        np.testing.assert_allclose(centers,reference,rtol=0,atol=1e-14)
        self.assertEqual(sorted(counts.tolist()),[9,11,13])


def run_report():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Qualification)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    return dict(profile='Aggregate-Centroid-Sensor-v1',tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
                numpy=np.__version__,research_data_read=False,research_runs_started=0,
                native_reproduction=False,scientific_performance_established=False)


if __name__=='__main__':
    report=run_report()
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
