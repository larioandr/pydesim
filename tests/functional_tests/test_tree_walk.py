from collections import namedtuple

import numpy as np

from pydesim import simulate


#############################################################################
# Tree model
#
# In this model we define a tree with each node having label and delay.
# During the model execution, we traverse a tree and put labels into trace
# list. Depending on the handlers selected, we either print labels on node
# enter, or on leave.
#
# The test tree looks like the following:
#
#       [A] delay=1.0
#        |
#       [B] delay=2.0
#        |
#     +--+--------------+
#     |                 |
#    [C] delay=3.0     [E] delay=5.0
#     |
#    [D] delay=4.0
#
class TreeNode:
    def __init__(self, delay, label):
        self.delay, self.label = delay, label
        self.visited = False
        self.parent = None
        self.children = []

    def reset(self):
        self.visited = False
        for child in self.children:
            child.reset()

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

    def find(self, label):
        ret = self if self.label == label else None
        index = 0
        while ret is None and index < len(self.children):
            ret = self.children[index].find(label)
            index += 1
        return ret

    def get(self, label):
        # noinspection PyNoneFunctionAssignment
        node = self.find(label)
        if node is None:
            raise ValueError(f'node "{label}" not found')
        return node

    def apply(self, fn, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        fn(self, *args, **kwargs)
        for child in self.children:
            child.apply(fn, args, kwargs)


def create_tree():
    """Construct a sample tree like (A, (B, (C, (D)), (E)))) with delays=1..5.
    """
    root = TreeNode(1.0, 'A')
    node_b = TreeNode(2.0, 'B')
    node_c = TreeNode(3.0, 'C')
    node_d = TreeNode(4.0, 'D')
    node_e = TreeNode(5.0, 'E')
    root.add_child(node_b)
    node_b.add_child(node_c)
    node_c.add_child(node_d)
    node_b.add_child(node_e)
    return root


class TreeData:
    """TreeData represents the model data used in subsequent tests.
    """
    tree_builder = create_tree  # a tree construction method

    def __init__(self, tree):
        """Initialize the Tree model data.

        :param tree: a tree (`TreeNode` instance) to be used in simulation.
        """
        self.tree = tree
        self.trace = []

    @classmethod
    def create(cls):
        tree = cls.tree_builder()
        return TreeData(tree)


def enter_node(sim, node):
    """Add label to trace, mark as visited, schedule enter to the first child.
    """
    assert not node.visited
    node.visited = True
    sim.data.trace.append(node.label)
    sim.schedule(node.delay, visit_next_child, args=(node,))


def leave_node(sim, node):
    """Schedule the next child visit for the parent node (if exists).
    """
    # We check visited to stop traversing parents when simulation started
    # not from the root node:
    if node.parent and node.parent.visited:
        sim.schedule(0, visit_next_child, args=(node.parent,))


def visit_next_child(sim, node):
    """Schedule enter to the first child that is not visited yet.
    """
    found_child = False
    for child in node.children:
        if not child.visited:
            sim.schedule(0, sim.handlers.get('enter'), args=(child,))
            found_child = True
            break
    if not found_child:
        sim.schedule(0, sim.handlers.leave, args=(node,))


def start_from_root(sim):
    """Default initialization method. Schedules enter to root node.
    """
    root = sim.data.tree
    sim.schedule(0, sim.handlers['enter'], kwargs={
        'node': root,
    })


###########################
# Tests for tree model

def test_tree_traverse_with_trace_on_enter_and_on_leave():
    """Validate traversing the tree twice writing trace on enter and on leave.

    In this test we call the simulation two times. For the first time, we
    use the standard traversing with adding node labels to trace on node enter.
    For the second time, we change enter and leave implementation to add node
    labels to trace during node leaving instead of entering. We check that the
    traversal time is the same (we still visit all nodes), but the trace
    is another.

    Besides that, we validate the results co-exist and are not shared.
    """
    def trace_on_leave__enter(sim, node):
        assert not node.visited
        node.visited = True
        sim.schedule(node.delay, visit_next_child, args=[node])

    def trace_on_leave__leave(sim, node):
        sim.data.trace.append(node.label)
        if node.parent and node.parent.visited:
            sim.schedule(0, visit_next_child, args=(node.parent,))

    trace_on_enter_result = simulate(TreeData, init=start_from_root, handlers={
        'enter': enter_node,
        'leave': leave_node,
    })

    trace_on_leave_result = simulate(TreeData, init=start_from_root, handlers={
        'enter': trace_on_leave__enter,
        'leave': trace_on_leave__leave,
    })

    trace_stime_limit_result = simulate(
        TreeData, init=start_from_root, handlers={
            'enter': enter_node,
            'leave': leave_node,
        }, stime_limit=7)

    np.testing.assert_allclose(trace_on_enter_result.stime, 15.0, atol=0.1)
    np.testing.assert_allclose(trace_on_leave_result.stime, 15.0, atol=0.1)
    np.testing.assert_allclose(trace_stime_limit_result.stime, 10.0, atol=0.1)
    assert trace_on_enter_result.data.trace == ['A', 'B', 'C', 'D', 'E']
    assert trace_on_leave_result.data.trace == ['D', 'C', 'E', 'B', 'A']
    assert trace_stime_limit_result.data.trace == ['A', 'B', 'C', 'D']


def test_tree_traverse_from_b_and_reverse_trace():
    """In this test we check that init and finalize methods are really called.

    We start the test at node B instead of A (root) and make sure that both
    trace and traversal time change. Moreover, we reverse the trace in the
    end by passing a finalization function which calls `list.reverse()`
    """
    def start_from_b(sim):
        root = sim.data.tree.find('B')
        sim.schedule(0, sim.handlers['enter'], args=[root])

    def reverse_trace(sim):
        sim.data.trace.reverse()

    ret = simulate(TreeData, init=start_from_b, fin=reverse_trace, handlers={
        'enter': enter_node,
        'leave': leave_node,
    })

    np.testing.assert_allclose(ret.stime, 14.0, atol=0.1, rtol=0.01)
    assert ret.data.trace == ['E', 'D', 'C', 'B']


def test_tree_traverse_in_parallel():
    """Validate events scheduled in parallel work properly, and using raw data.

    In this test we schedule visiting all children once we enter the state,
    after the same delay. We do not come back to parent, so 'leave' event
    is not specified. This is like traversing the tree using unlimited number
    of threads by starting new thread for each child just after processing
    the parent. Since then, the delay is just the maximum delay of a branch.

    Besides that, we define data as a `namedtuple`, not `TreeData`, and check
    that is is used properly and put into the result.

    To be sure the events are queued in the correct order, we add several
    nodes (F, G, H, I, J) to the tree:

    (A [d=1], (B [d=2], (C [d=3], (D [d=4], (J [d=1]))),
                        (E [d=5], (H [d=1], (I [d=1]))),
                        (F [d=2], (G [d=2]))
               )
    )

    The visiting order is expected like this:
    A -> B -> C -> E -> F -> G -> D -> H -> I -> J
    """
    tree = create_tree()
    node_f = TreeNode(2.0, 'F')
    node_g = TreeNode(2.0, 'G')
    node_h = TreeNode(1.0, 'H')
    node_i = TreeNode(1.0, 'I')
    node_j = TreeNode(1.0, 'J')

    tree.get('B').add_child(node_f)
    tree.get('D').add_child(node_j)
    tree.get('E').add_child(node_h)
    node_f.add_child(node_g)
    node_h.add_child(node_i)

    data_class = namedtuple('ModelData', ['tree', 'trace'])
    data = data_class(tree=tree, trace=[])

    def visit(sim, node):
        assert not node.visited
        node.visited = True
        sim.data.trace.append(node.label)
        if node.children:
            for child in node.children:
                sim.schedule(node.delay, visit, args=[child])
        else:
            sim.schedule(node.delay)

    ret = simulate(data, init=start_from_root, handlers={'enter': visit})

    np.testing.assert_allclose(ret.stime, 11.0, atol=0.1, rtol=0.01)
    assert ret.data.trace == ['A', 'B', 'C', 'E', 'F', 'G', 'D', 'H', 'I', 'J']
    assert ret.data == data  # also check that data is the same


def test_cancel_tree_printing_before_deadline():
    """Validate cancel operation works properly.

    In this test we schedule printing all the nodes ('A'..'E') at one moment
    (at t=5). However, we also launch tree traversal (from t=0), and we
    cancel the scheduled print when visiting the tree.

    It looks like we will cancel printing for nodes A, B and С. When printing,
    we also cancel entering the node, so simulation will stop at t=5, after
    print all nodes to which we didn't come.
    """
    def enter_silent(sim, node):
        sim.cancel(sim.data.visit_evids[node.label])
        del sim.data.visit_evids[node.label]
        for child in node.children:
            eid = sim.schedule(node.delay, enter_silent, args=(child,))
            sim.data.enter_evids[child.label] = eid

    def visit_node(sim, node):
        sim.data.trace.append(node.label)
        sim.cancel(sim.data.enter_evids[node.label])
        del sim.data.enter_evids[node.label]

    def init(sim):
        def schedule_visit(node):
            eid = sim.schedule(5, visit_node, args=(node,))
            sim.data.visit_evids[node.label] = eid

        sim.data.tree.apply(schedule_visit)
        sim.schedule(0, enter_silent, args=(sim.data.tree,))

    data_class = namedtuple(
        'ModelData', ['tree', 'trace', 'visit_evids', 'enter_evids']
    )
    data = data_class(create_tree(), trace=[], visit_evids={}, enter_evids={})

    ret = simulate(data, init)

    assert ret.data.trace == ['D']
    assert ret.stime == 5
