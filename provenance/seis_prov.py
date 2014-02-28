import cgi
import collections
from lxml import etree
import pydot
import re
from uuid import uuid4
import warnings

from graph import *


# The used namespaces
NS_PROV = "http://www.w3.org/ns/prov#"
NS_SEIS_PROV = "http://sdf.readthedocs.org"


NSMAP = {
    "prov": NS_PROV,
    "seis_prov": NS_SEIS_PROV
}

NS_MAP_REVERSE = {value: key for (key, value) in NSMAP.items()}


# The allowed types of connections.
CONNECTION_TYPES = ("wasGeneratedBy", "used", "wasAssociatedWith",
                    "actedOnBehalfOf")

def _get_tag(name, namespace=None):
    if not namespace:
        return name
    return "{%s}%s" % (namespace, name)


class SeisProvGraph(object):
    """
    Class used to represent seismological provenance graphs. The used
    notation is the one defined by W3C PROV.
    """
    def __init__(self):
        self.__nodes = []
        self.__edges = []

    def add_node(self, node):
        """
        Adds a node to the graph.
        """
        if node in self.__nodes:
            msg = "Node '%s' already in graph. Will not be added again."
            warnings.warn(msg)
            return
        existing_ids = [_i.id for _i in self.__nodes]
        if node.id in existing_ids:
            msg = "A node with this id already exists in the graph."
            raise ValueError(msg)
        self.__nodes.append(node)

    def add_edge(self, edge):
        """
        Adds an edge to the graph. Both ends of the edge must already exist
        in the graph.
        """
        if edge.origin not in self.__nodes:
            msg = "Origin of edge not contained in graph."
            raise ValueError(msg)
        elif edge.destination not in self.__nodes:
            msg = "Destination of edge not contained in graph."
            raise ValueError(msg)
        if edge in self.__edges:
            msg = "Edge '%s' already in graph. Will not be added again."
            warnings.warn(msg)
            return
        self.__edges.append(edge)

    def create_and_add_edge(self, origin, destination, connection_type):
        edge = Connection(connection_type, origin, destination)
        self.add_edge(edge)

    def process_waveform_entity(self, waveform_entity, processing_type,
                                processing_options,
                                processing_agent=None):
        processing_node = SeismicProcessingActivity(processing_type,
                                                    processing_options)

        new_entity = WaveformDataEntity()
        self.add_node(processing_node)
        self.add_node(new_entity)

        # Add the agent if given.
        if processing_agent:
            self.create_and_add_edge(processing_node, processing_agent,
                                     "wasAssociatedWith")

        # Connect the activity with the entities.
        self.create_and_add_edge(processing_node, waveform_entity, "used")
        self.create_and_add_edge(new_entity, processing_node,
                                 "wasGeneratedBy")

        return new_entity

    def plot(self, filename=None):
        # Collect already plotted instance here to avoid circular runs in the
        # graph.
        graph = pydot.Dot("graphname", graph_type="digraph", resolution=150,
                          rankdir="BT")
        pydot_nodes = {}
        for node in self.__nodes:
            pydot_nodes[node] = pydot.Node(node.id, label=node.label,
                                           **node.get_style())
            graph.add_node(pydot_nodes[node])

            info = node._get_info()

            if info:
                # Add an annotations box to the plot
                ann_rows = [ANNOTATION_START_ROW]
                ann_rows.extend(
                    ANNOTATION_ROW_TEMPLATE % (cgi.escape(unicode(attr)),
                                               cgi.escape(unicode(value[0])))
                    for attr, value in info.items())
                ann_rows.append(ANNOTATION_END_ROW)
                annotations = pydot.Node(
                    'ann_%s' % str(uuid4()),
                    label='\n'.join(ann_rows),
                    **ANNOTATION_STYLE)
                graph.add_node(annotations)
                graph.add_edge(pydot.Edge(annotations, pydot_nodes[node],
                                          **ANNOTATION_LINK_STYLE))

        for edge in self.__edges:
            this_connection = pydot.Edge(
                pydot_nodes[edge.origin],
                pydot_nodes[edge.destination],
                **edge.get_style())
            graph.add_edge(this_connection)


        if filename is None:
            import io
            from PIL import Image

            temp = io.BytesIO(graph.create_png())

            Image.open(temp).show()
            temp.close()
        elif filename.endswith(".png"):
            with open(filename, "wb") as fh:
                fh.write(graph.create_png())
        elif filename.endswith(".svg"):
            with open(filename, "wb") as fh:
                fh.write(graph.create_svg())
        elif filename.endswith(".pdf"):
            with open(filename, "wb") as fh:
                fh.write(graph.create_pdf())
        else:
            raise NotImplementedError

    def toXML(self):
        xml_elements = {}
        doc = etree.Element(_get_tag("document", NS_PROV), nsmap=NSMAP)

        for node in self.__nodes:
            elem = etree.SubElement(doc, node._get_xml_tag())
            xml_elements[node] = elem
            elem.set(_get_tag("id", NS_PROV), node.id)

            label = etree.SubElement(elem, _get_tag("label", NS_PROV))
            label.text = node.label


            for tag, info in node._get_info().items():
                text, ns = info
                subel = etree.SubElement(elem, _get_tag(tag, ns))
                subel.text = text

        for edge in self.__edges:
            elem = etree.SubElement(doc, edge._get_xml_tag())

            origin_elem = etree.SubElement(elem, edge.origin._get_xml_tag())
            origin_elem.set(_get_tag("ref", NS_PROV), edge.origin.id)

            destination_elem = etree.SubElement(
                elem, edge.destination._get_xml_tag())
            destination_elem.set(_get_tag("ref", NS_PROV), edge.destination.id)


        print(etree.tostring(doc, pretty_print=True, xml_declaration=True,
                             encoding="UTF-8"))


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

    def _get_xml_tag(self):
        return _get_tag(self.connection_type, NS_PROV)



class Node(object):
    """
    A generic node object in the graph.

    A node is identified by its id which must be unique in each graph. The
    uniqueness must be guaranteed by some outside logic.

    Each Node futhermore has inbound and outbound connections.
    """
    def __init__(self, id, label=None,
                 in_connections=None, out_connections=None):
        self.id = id
        self.label = label or id
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

    def _get_info(self):
        return {}

    def add_agent(self, Agent, connection_type):
        """
        Add an agent to this node.
        """
        self.connect_to(Agent, connection_type)

    def _get_xml_tag(self):
        raise NotImplementedError



class Entity(Node):
    """
    Class representing a W3C Prov entity. It has a name and a specific
    entity type which is meant to be defined by a subclass.
    """
    def __init__(self, id, label):
        super(Entity, self).__init__(id=id, label=label)

    def derive_new_entity_with_activity(self, activity, activity_options):
        """
        Function creating a new entity of the same type by applying some
        activity with a number of options.

        Returns the new entity.
        """
        new_entity = self.__class__(id=self.id)
        if not hasattr(new_entity, "entity_type") \
                or not new_entity.entity_type:
            new_entity.entity_type = self.entity_type
        activity = Activity(activity, activity_options)
        self.connect_from(activity, "used")
        activity.connect_from(new_entity, "wasGeneratedBy")
        return new_entity, activity

    def get_style(self):
        return PROV_REC_ENTITY

    def _get_xml_tag(self):
        return _get_tag("entity", NS_PROV)


class Activity(Node):
    """
    Class representing a W3C PROV Activity.
    """
    def __init__(self, id, label=None):
        super(Activity, self).__init__(id=id, label=label)

    def get_style(self):
        return PROV_REC_ACTIVITY


class SeismicProcessingActivity(Activity):
    """
    Class representing a seismic processing activity.
    """
    def __init__(self, processing_type, processing_options, id=None):
        if id is None:
            id = "seismic_processing_%s_%s" % (
                processing_type.lower().replace(" ", "_"),
                str(uuid4()))
        super(SeismicProcessingActivity, self).__init__(id=id,
                                                        label=processing_type)
        self.processing_options = processing_options

    def _get_info(self):
        info = collections.OrderedDict()
        for key, value in self.processing_options.items():
            info[key] = (value, NS_SEIS_PROV)
        return info

    def _get_xml_tag(self):
        return _get_tag("seismicProcessing", NS_SEIS_PROV)


class Agent(Node):
    """
    Class representing a W3C PROV Agent.
    """
    def __init__(self, id, label=None):
        super(Agent, self).__init__(id=id, label=label)

    def get_style(self):
        return PROV_REC_AGENT


class SoftwareAgent(Agent):
    def __init__(self, name, version, url, id=None):
        if not id:
            id = "%s_%s_%s" % (name.lower().replace(" ", "_"),
                               version.lower().replace(" ", "_"),
                               str(uuid4()))

        super(SoftwareAgent, self).__init__(
            id=id, label="%s %s" % (name, version))
        self.name = name
        self.version = version
        self.url = url

    def _get_xml_tag(self):
        return _get_tag("softwareAgent", NS_PROV)

    def _get_info(self):
        info = collections.OrderedDict()
        info["softwareName"] = (self.name, NS_SEIS_PROV)
        info["softwareVersion"] = (self.version, NS_SEIS_PROV)
        info["URL"] = (self.url, NS_SEIS_PROV)
        return info


class Person(Agent):
    def __init__(self, name, email=None, institution=None):
        id = name.lower().replace(" ", "_") + ("_%s" % str(uuid4()))
        super(Person, self).__init__(id=id, label=name)
        self.email = email
        self.institution = institution

    def _get_info(self):
        info = collections.OrderedDict()
        if self.email:
            info["eMail"] = (self.email, NS_SEIS_PROV)
        if self.institution:
            info["institution"] = (self.institution, NS_SEIS_PROV)
        return info

    def _get_xml_tag(self):
        return _get_tag("person", NS_PROV)


class WaveformDataEntity(Entity):
    def __init__(self, id=None):
        if id is None:
            id = "waveform_data_%s" % str(uuid4())
        super(WaveformDataEntity, self).__init__(id, label="Waveform Data")

    def _get_xml_tag(self):
        return _get_tag("waveformDataEntity", NS_SEIS_PROV)


class ConfigFileEntity(Entity):
    def __init__(self, filename, content=None, id=None):
        if id is None:
            id = "config_file_%s" % str(uuid4())
        super(ConfigFileEntity, self).__init__(id, label="Config File")
        self.filename = filename
        self.content = content

    def _get_xml_tag(self):
        return _get_tag("configFile", NS_SEIS_PROV)

    def _get_info(self):
        info = collections.OrderedDict()
        info["filename"] = (self.filename, NS_SEIS_PROV)
        if self.content:
            info["fileContent"] = (self.content[:20] + "...", NS_SEIS_PROV)
        return info


class EarthModelEntity(Entity):
    def __init__(self, model_name, description=None, id=None):
        if id is None:
            id = "earth_model_%s_%s" % (model_name, str(uuid4()))
        super(EarthModelEntity, self).__init__(id, label="Earth Model")
        self.model_name = model_name
        self.description = description

    def _get_xml_tag(self):
        return _get_tag("earthModel", NS_SEIS_PROV)

    def _get_info(self):
        info = collections.OrderedDict()
        info["modelName"] = (self.model_name, NS_SEIS_PROV)
        info["description"] = (self.description, NS_SEIS_PROV)
        return info


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

def trace_to_graph(trace, name, email, institution):
    if not hasattr(trace.stats, "processing"):
        msg = "Empty processing info for the trace."
        raise ValueError(msg)

    steps = []
    for step in trace.stats.processing:
        software = step[:step.index(":")]
        command = step[step.index(":") + 1:]
        sofware, sofware_version = [_i.strip() for _i in software.split()]
        regex = re.compile("(.*)\((.*)\)")
        command_name, arguments = re.match(regex, command).groups()
        arguments = arguments.split("::")
        args = {}
        for arg in arguments:
            i, j = arg.split("=")
            args[i] = j
        steps.append((command_name, args))


    obspy = SoftwareAgent(name=sofware, version=sofware_version,
                          url="http://www.obspy.org")
    person = Person(name=name, email=email, institution=institution)

    graph = SeisProvGraph()
    graph.add_node(obspy)
    graph.add_node(person)
    graph.create_and_add_edge(obspy, person, "actedOnBehalfOf")

    data = WaveformDataEntity()
    graph.add_node(data)

    for step in steps:
        data = graph.process_waveform_entity(data, step[0], step[1], obspy)

    graph.plot()





if __name__ == "__main__":
    import obspy
    tr = obspy.read()[0]
    tr.detrend("linear")
    tr.filter("lowpass", freq=2.0)
    tr.decimate(2)
    tr.integrate()

    trace_to_graph(tr, "Lion Krischer",
                   "krischer[at]geophysik.uni-muenchen.de",
                   "LMU")
    import sys
    sys.exit()























    # # Initialize a new graph.
    # graph = SeisProvGraph()
    # # Create a person, a provenance agent.
    # lion = Person(name="Lion Krischer",
    #               email="krischer[at]geophysik.uni-muenchen.de",
    #               institution="LMU")
    # graph.add_node(lion)
    #
    # # Create a software package, a provenance agent.
    # obspy = SoftwareAgent(name="ObsPy", version="0.9.0",
    #                       url="http://www.obspy.org")
    # graph.add_node(obspy)
    #
    # # ObsPy was steered by a person.
    # graph.create_and_add_edge(obspy, lion, "actedOnBehalfOf")
    #
    # data_1 = WaveformDataEntity()
    # graph.add_node(data_1)
    #
    # data_2 = graph.process_waveform_entity(
    #     data_1, "detrend", {"type":  "linear"}, obspy)
    #
    # data_3 = graph.process_waveform_entity(data_2, "lowpass_filter",
    #                               {"type":  "Butterworth",
    #                                "frequency": "2.0",
    #                                "order": "2"}, obspy)
    #
    # data_4 = graph.process_waveform_entity(data_3, "decimate",
    #                                        {"factor":  "4"}, obspy)
    #
    # graph.toXML()
    # graph.plot()

    # Initialize a new graph.
    graph = SeisProvGraph()
    # Create a person, a provenance agent.
    lion = Person(name="Lion Krischer",
                  email="krischer[at]geophysik.uni-muenchen.de",
                  institution="LMU")
    james = Person(name="James Smith",
                  email="jas11[at]princeton.edu",
                  institution="Princeton")
    graph.add_node(lion)
    graph.add_node(james)

    # Create a software package, a provenance agent.
    specfem = SoftwareAgent(
        name="SPECFEM3D GLOBE", version="5.1.5",
        url="http://geodynamics.org/cig/software/specfem3d_globe/")
    graph.add_node(specfem)

    graph.create_and_add_edge(specfem, lion, "actedOnBehalfOf")

    data_1 = WaveformDataEntity()
    graph.add_node(data_1)

    simulation = SeismicProcessingActivity("Waveform Simulation", {})
    graph.add_node(simulation)

    earth_model = EarthModelEntity(model_name="NorthAtlantic",
                                   description="Some random model.")

    graph.add_node(earth_model)

    config_file_1 = ConfigFileEntity(filename="Par_file", content="""
PDE 2012 4 12 7 15 48.50 39.26000 41.04000 5.00000 4.7 4.7 2012-04-12T07:15:48.500000Z_4.7
event name:      0000000
time shift:       0.0000
half duration:    0.0000
latitude:       39.26000
longitude:      41.04000
depth:          5.00000
Mrr:         1e+23
Mtt:         1e+23
Mpp:         1e+23
Mrt:         0
Mrp:         0
Mtp:         0""".strip())
    graph.add_node(config_file_1)
    graph.create_and_add_edge(simulation, config_file_1,  "used")
    config_file_2 = ConfigFileEntity(filename="CMTSOLUTION")
    graph.add_node(config_file_2)
    graph.create_and_add_edge(simulation, config_file_2,  "used")
    config_file_3 = ConfigFileEntity(filename="STATIONS")
    graph.add_node(config_file_3)
    graph.create_and_add_edge(simulation, config_file_3,  "used")

    graph.create_and_add_edge(earth_model, james, "wasAssociatedWith")

    graph.create_and_add_edge(simulation, earth_model, "used")
    graph.create_and_add_edge(data_1, simulation, "wasGeneratedBy")
    graph.create_and_add_edge(simulation, specfem, "wasAssociatedWith")


    graph.toXML()
    graph.plot("simulation.svg")
