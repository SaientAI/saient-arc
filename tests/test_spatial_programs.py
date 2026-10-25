import unittest
from arc_gremlin.spatial_programs import (perceive, Selector, Operation, Program,
                                         execute, synthesize, agreement)


def pair(a,b): return {'input':a,'output':b}


# Six independently specified tasks: two demonstrations and an unseen input.
TASKS = {
 'crop': [pair([[0,0,0,0],[0,2,2,0],[0,0,0,3]],[[2,2]]),
          pair([[0,4,0,0],[0,4,0,1],[0,0,0,0]],[[4],[4]]),
          pair([[0,0,0,0],[0,6,6,6],[1,0,0,0]],[[6,6,6]])],
 'keep': [pair([[2,2,0],[0,0,3]],[[2,2,0],[0,0,0]]),
          pair([[0,4,0],[0,4,0],[1,0,0]],[[0,4,0],[0,4,0],[0,0,0]]),
          pair([[0,6,6],[1,0,0]],[[0,6,6],[0,0,0]])],
 'recolor': [pair([[2,2,0],[0,0,3]],[[7,7,0],[0,0,3]]),
             pair([[0,4,0],[0,4,0],[1,0,0]],[[0,7,0],[0,7,0],[1,0,0]]),
             pair([[0,6,6],[1,0,0]],[[0,7,7],[1,0,0]])],
 'translate': [pair([[2,2,0,0],[0,0,0,3]],[[0,2,2,0],[0,0,0,3]]),
               pair([[4,0,0,0],[4,0,0,1]],[[0,4,0,0],[0,4,0,1]]),
               pair([[0,6,6,0],[1,0,0,0]],[[0,0,6,6],[1,0,0,0]])],
 'holes': [pair([[2,2,2],[2,0,2],[2,2,2]],[[2,2,2],[2,7,2],[2,2,2]]),
           pair([[4,4,4,4],[4,0,0,4],[4,4,4,4]],[[4,4,4,4],[4,7,7,4],[4,4,4,4]]),
           pair([[6,6,6],[6,0,6],[6,0,6],[6,6,6]],[[6,6,6],[6,7,6],[6,7,6],[6,6,6]])],
 'compose': [pair([[0,0,0,0],[0,2,2,0],[0,0,0,3]],[[7,7]]),
             pair([[0,4,0,0],[0,4,0,1],[0,0,0,0]],[[7],[7]]),
             pair([[0,0,0,0],[0,6,6,6],[1,0,0,0]],[[7,7,7]])],
}


class SpatialTests(unittest.TestCase):
    def test_six_synthesis_tasks(self):
        for name, examples in TASKS.items():
            with self.subTest(name=name):
                result = synthesize(examples[:2],max_candidates=1600,budget_ms=800)
                self.assertIsNotNone(result['program'])
                self.assertEqual(result['train_score'][0],2,result)
                self.assertTrue(result['holdout'])
                self.assertEqual(execute(result['program'],examples[2]['input']),examples[2]['output'])
                self.assertLessEqual(result['attempts'],1600)
                if name == 'compose': self.assertEqual(len(result['program'].operations),2)

    def test_scene_graph(self):
        scene = perceive([[2,2,2,2,2],[2,0,0,0,2],[2,0,3,0,2],[2,0,0,0,2],[2,2,2,2,2]])
        self.assertEqual(scene.objects[0].bbox,(0,0,4,4))
        self.assertEqual(scene.objects[0].centroid,(2,2))
        self.assertEqual(scene.objects[1].mask,((0,0),))
        self.assertIn((0,'bbox_contains',1),[(r.source,r.kind,r.target) for r in scene.relations])
        scene = perceive([[1,2,0,3],[0,0,0,0]])
        kinds = {(r.source,r.kind,r.target) for r in scene.relations}
        self.assertIn((0,'touching4',1),kinds)
        self.assertIn((0,'left_of',2),kinds)
        self.assertIn((0,'same_shape',2),kinds)
        self.assertEqual(len(perceive([[1,0],[0,1]]).objects),2)
        self.assertEqual(len(perceive([[1,0],[0,1]],8).objects),1)

    def test_transforms_and_relational_selection(self):
        def run(kind,args,g,selector=Selector()):
            return execute(Program((Operation(kind,selector,args),)),g)
        self.assertEqual(run('reflect',(1,),[[2,0],[2,2]]),[[0,2],[2,2]])
        self.assertEqual(run('rotate',(1,),[[2,0],[2,2]]),[[2,2],[2,0]])
        self.assertEqual(run('fill_rectangle',(7,),[[2,0],[2,2]]),[[7,7],[7,7]])
        self.assertEqual(run('recolor',(7,),[[1,2,0,3]],Selector('touching')),[[7,7,0,3]])
        with self.assertRaises(ValueError): run('translate',(0,1),[[2,2]])
        with self.assertRaises(ValueError): run('translate',(0,1),[[2,2,3]])

    def test_safety_and_validation(self):
        with self.assertRaises(ValueError): Program((Operation('python'),))
        with self.assertRaises(ValueError): Program((Operation('crop'),)*3)
        with self.assertRaises(ValueError): perceive([[1],[1,2]])
        self.assertEqual(agreement([[[1]]],[pair([[0]],[[1,1]])]),(0,0.0))
        result = synthesize(TASKS['crop'][:2],max_candidates=0)
        self.assertEqual(result['attempts'],0)
        self.assertIsNone(result['program'])

    def test_holdout_is_not_search_feedback(self):
        train = TASKS['crop'][:2]
        a = synthesize(train)
        b = synthesize([train[0],pair(train[1]['input'],[[9]])])
        self.assertEqual(a['program'],b['program'])
        self.assertEqual(a['attempts'],b['attempts'])
        self.assertFalse(b['holdout'])


if __name__ == '__main__': unittest.main()
