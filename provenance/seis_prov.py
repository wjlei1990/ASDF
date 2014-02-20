import collections
import pydot

from graph import *

# The namespace used in the XML representation.
NAMESPACE = ("seis_prov", "http://sdf.readthedocs.org")
# The allowed types of connections.
CONNECTION_TYPES = ("wasGeneratedBy", "used", "wasAssociatedWith",
                    "actedOnBehalfOf")


class Connection(object):
    """
    A connection or edge in a graph. This is a directed connection, i.e. it
    has an origin and a destination. As it is meant to represent provenance
    information it will always point towards the past.
    """
    def __init__(self, connection_type, origin, destination):
        """
        :param connection_type: The type of connection as defined in W3C PROV.
        :param origin: The origin of the connection. This will be the
            current time.
        :param destination: The destination of the connection. Will point
            towards the past.
        """
        if connection_type not in CONNECTION_TYPES:
            msg = "Connection type '%s' invalid. Valid types: %s" % (
                connection_type, ", ".join(CONNECTION_TYPES))
            raise ValueError(msg)
        self.connection_type = connection_type
        self.origin = origin
        self.destination = destination

    def get_style(self):
        """
        Returns the pydot style for the given connection.
        """
        if self.connection_type == "wasGeneratedBy":
            return PROV_REC_GENERATION
        elif self.connection_type == "used":
            return PROV_REC_USAGE
        elif self.connection_type == "wasAssociatedWith":
            return PROV_REC_ASSOCIATION
        elif self.connection_type == "actedOnBehalfOf":
            return PROV_REC_DELEGATION
        else:
            raise NotImplementedError


class Node(object):
    """
    A generic node object in the graph. Each node can have an arbitrary
    number of in- and out connections. In connections come from the future
    and out connections point towards the past.
    """
    def __init__(self, name, in_connections=None, out_connections=None):
        self.name = name
        self.in_connections = []
        self.out_connections = []
        if in_connections:
            self.in_connectisns.extend(in_connections)
        if out_connections:
            self.out_connectisns.extend(out_connections)

    def connect_to(self, other_node, connection_type):
        """
        Create a new connection with a node in this objects past.

        :param other_node: The node which represents part of the history of
            the current node.
        :param connection_type: The type of connection to be created.
        """
        connection = Connection(connection_type, self, other_node)
        self.out_connections.append(connection)
        other_node.in_connections.append(connection)

    def connect_from(self, other_node, connection_type):
        """
        Create a new connection with a node in this objects future.

        :param other_node: The node which represents the future of the
            current node.
        :param connection_type: The type of connection to be created.
        """
        connection = Connection(connection_type, other_node, self)
        self.in_connections.append(connection)
        other_node.out_connections.append(connection)

    def _get_bundle(self):
        return None

    @property
    def bundle(self):
        bundle = self._get_bundle()
        if not bundle:
            return None
        new_bundle = collections.OrderedDict()
        for key, value in bundle.items():
            new_bundle["%s:%s" % (NAMESPACE[0], key)] = value
        return new_bundle

    def add_agent(self, Agent, connection_type):
        """
        Add an agent to this node.
        """
        self.connect_to(Agent, connection_type)



class Entity(Node):
    """
    Class representing a W3C Prov entity. It has a name and a specific
    entity type which is meant to be defined by a subclass.
    """
    def __init__(self, name, entity_type=None):
        super(Entity, self).__init__(name=name)
        self.entity_type = entity_type

    def generate_new_data_with_activity(self, activity, activity_options):
        """
        Function creating a new entity of the same type by applying some
        activity with a number of options.

        Returns the new entity.
        """
        new_entity = self.__class__(name=self.name)
        if not hasattr(new_entity, "entity_type") \
                or not new_entity.entity_type:
            new_entity.entity_type = self.entity_type
        activity = Activity(activity, activity_options)
        self.connect_from(activity, "used")
        activity.connect_from(new_entity, "wasGeneratedBy")
        return new_entity, activity

    def get_style(self):
        return PROV_REC_ENTITY



class Activity(Node):
    """
    Class representing a W3C PROV Activity.
    """
    def __init__(self, name, activity_options):
        super(Activity, self).__init__(name=name)
        self.activity_options = activity_options

    def get_style(self):
        return PROV_REC_ACTIVITY

    def _get_bundle(self):
        return self.activity_options


class Agent(Node):
    """
    Class representing a W3C PROV Agent.
    """
    def __init__(self, name):
        super(Agent, self).__init__(name=name)

    def get_style(self):
        return PROV_REC_AGENT


class Software(Agent):
    def __init__(self, name, version, url=None, description=None,
                 command=None):
        super(Software, self).__init__(name=name)
        self.version = version
        self.url = url
        self.description = description

    def _get_bundle(self):
        bundle = collections.OrderedDict()
        bundle["version"] = self.version
        if self.url:
            bundle["URL"] = self.url
        if self.description:
            bundle["description"] = self.description
        return bundle


class Person(Agent):
    def __init__(self, name, email=None, institution=None):
        super(Person, self).__init__(name=name)
        self.email = email
        self.institution = institution

    def _get_bundle(self):
        bundle = collections.OrderedDict()
        if self.email:
            bundle["E-Mail"] = self.email
        if self.institution:
            bundle["Institution"] = self.institution
        return bundle


class WaveformData(Entity):
    def __init__(self, name):
        super(WaveformData, self).__init__(
            name=name, entity_type="seismic_waveform")


def plot(entity):
    """
    Plots all connection coming in here.
    """
    # Collect already plotted instance here to avoid circular runs in the
    # graph.
    pydot_nodes = {}

    graph = pydot.Dot("graphname", graph_type="digraph", resolution=150,
                      rankdir="BT")
    graph.counter = collections.Counter()

    plot_entity(entity, graph, pydot_nodes)

    import io
    from PIL import Image

    temp = io.BytesIO(graph.create_png())

    Image.open(temp).show()
    temp.close()


def plot_entity(entity, graph, pydot_nodes):
    """
    This recursive functions walks the graph starting at any entity and plots
    all nodes and edges with the help of pydot/graphviz.
    """
    import random
    blub = random.randint(0, 1231231312)

    # If necessary, first add any nodes.
    if id(entity) in pydot_nodes:
        this_node = pydot_nodes[id(entity)]
        return this_node
    elif isinstance(entity, Entity):
        graph.counter["entity_" + entity.name] += 1
        this_node = pydot.Node(entity.entity_type + str(blub), label="%s\:%s" %
                (NAMESPACE[0], entity.name),
                 **PROV_REC_ENTITY)
        pydot_nodes[id(entity)] = this_node
        graph.add_node(this_node)
    elif isinstance(entity, Activity):
        graph.counter["activity_" + entity.name] += 1
        this_node = pydot.Node(str(blub), label="%s\:%s" %
                (NAMESPACE[0], entity.name),
                **PROV_REC_ACTIVITY)
        pydot_nodes[id(entity)] = this_node
        graph.add_node(this_node)
    elif isinstance(entity, Agent):
        graph.counter["agent_" + entity.name] += 1
        this_node = pydot.Node(str(blub),
                               label="%s\:%s" % (NAMESPACE[0], entity.name),
                               **PROV_REC_AGENT)
        pydot_nodes[id(entity)] = this_node
        graph.add_node(this_node)
    else:
        raise NotImplementedError

    if entity.bundle:
        import cgi

        ann_rows = [ANNOTATION_START_ROW]
        ann_rows.extend(ANNOTATION_ROW_TEMPLATE % \
            (cgi.escape(unicode(attr)), cgi.escape(unicode(value))) for
                        attr, value in entity.bundle.items())
        ann_rows.append(ANNOTATION_END_ROW)
        annotations = pydot.Node('ann_%i' % blub, label='\n'.join(ann_rows),
                                 **ANNOTATION_STYLE)
        graph.add_node(annotations)
        graph.add_edge(pydot.Edge(annotations, this_node,
                                  **ANNOTATION_LINK_STYLE))


    # Otherwise loop over all connections.
    for connection in entity.in_connections:
        # Check if this connection has already been added!
        if id(connection) in pydot_nodes:
            connection_to_be_plotted = False
        else:
            connection_to_be_plotted = True
            pydot_nodes[id(connection)] = True

        other_node = plot_entity(connection.origin, graph, pydot_nodes)
        if not other_node:
            continue

        if connection_to_be_plotted is True:
            this_connection = pydot.Edge(other_node, this_node,
                                         **connection.get_style())
            graph.add_edge(this_connection)

    for connection in entity.out_connections:
        # Check if this connection has already been added!
        if id(connection) in pydot_nodes:
            connection_to_be_plotted = False
        else:
            connection_to_be_plotted = True
            pydot_nodes[id(connection)] = True

        other_node = plot_entity(connection.destination, graph, pydot_nodes)
        if not other_node:
            continue

        if connection_to_be_plotted is True:
            this_connection = pydot.Edge(this_node, other_node,
                                         **connection.get_style())
            graph.add_edge(this_connection)

    return this_node


if __name__ == "__main__":

    obspy = Software(name="ObsPy", version="0.9.0", url="http://obspy.org")
    lion = Person(name="Lion Krischer",
                  email="krischer[at]geophysik.uni-muenchen.de",
                  institution="LMU")
    obspy.add_agent(lion, "actedOnBehalfOf")

    data = WaveformData("waveform")
    data2, activity = data.generate_new_data_with_activity(
        "detrend",  {"type":  "linear"})
    data3, activity_2 = data2.generate_new_data_with_activity(
        "lowpass_filter", {"frequency": "2.0", "type": "Butterworth",
                           "order": "2"})
    data4, activity_3 = data3.generate_new_data_with_activity(
        "decimate", {"factor": "4"})

    activity.add_agent(obspy, "wasAssociatedWith")
    activity_2.add_agent(obspy, "wasAssociatedWith")
    activity_3.add_agent(obspy, "wasAssociatedWith")

    plot(data)
