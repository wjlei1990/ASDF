me = pr.agent("seis_prov:pp_me", other_attributes=(
    ("prov:type",
        prov.identifier.QualifiedName(prov.constants.PROV, "Person")),
    ("prov:label", "Hans Mustermann"),
    ("seis_prov:name", "Hans Mustermann"),
    ("seis_prov:email", "hans.mustermann@email.com")
))

other = pr.agent("seis_prov:pp_other", other_attributes=(
    ("prov:type",
        prov.identifier.QualifiedName(prov.constants.PROV, "Person")),
    ("prov:label", "Susanna Musterfrau"),
    ("seis_prov:name", "Susanna Musterfrau"),
    ("seis_prov:email", "susanna.musterfrau@email.com")
))

specfem = pr.agent("seis_prov:sa_9DIG8A-TA", other_attributes=(
    ("prov:type",
        prov.identifier.QualifiedName(prov.constants.PROV, "SoftwareAgent")),
    ("prov:label", "SPECFEM3D GLOBE"),
    ("seis_prov:software_name", "SPECFEM3D GLOBE"),
    ("seis_prov:software_version", "6.0.0"),
    ("seis_prov:website", "http://geodynamics.org/cig/software/specfem3d")
))

model = pr.entity("seis_prov:em_SKFUSJDOEJ", other_attributes=(
    ("prov:label", "Earth Model"),
    ("prov:type", "seis_prov:earth_model"),
    ("seis_prov:model_name", "Random Model"),
    ("seis_prov:website", "http://random.org/model")
))


param = pr.entity("seis_prov:ip_38JD89DA8L", other_attributes=(
    ("prov:label", "Input Parameter"),
    ("prov:type", "seis_prov:input_parameter"),
    ("seis_prov:model_name", "Random Model"),
    ("seis_prov:website", "http://random.org/model"),
    ("seis_prov:SIMULATION_TYPE", 1),
    ("seis_prov:NOISE_TOMOGRAPHY", 0),
    ("seis_prov:NCHUNKS", 1),
    ("seis_prov:ANGULAR_WIDTH_XI_IN_DEGREES", 90.0 ),
    ("seis_prov:ANGULAR_WIDTH_ETA_IN_DEGREES", 90.0),
    ("seis_prov:CENTER_LATITUDE_IN_DEGREES", 40.0),
    ("seis_prov:CENTER_LONGITUDE_IN_DEGREES", 10.0),
    ("seis_prov:GAMMA_ROTATION_AZIMUTH", 20.0),
    ("seis_prov:NEX_XI", 240),
    ("seis_prov:NEX_ETA", 240),
    ("seis_prov:NPROC_XI", 5),
    ("seis_prov:NPROC_ETA", 5),
    ("seis_prov:ANISOTROPIC_KL", False),
    ("seis_prov:RECEIVERS_CAN_BE_BURIED", True),
    ("seis_prov:USE_LDDRK", False),
    ("seis_prov:EXACT_MASS_MATRIX_FOR_ROTATION", False),
    ("seis_prov:ABSORBING_CONDITIONS", False),
    ("seis_prov:OCEANS", False),
    ("seis_prov:ELLIPTICITY", False),
    ("seis_prov:TOPOGRAPHY", False),
    ("seis_prov:GRAVITY", False),
    ("seis_prov:ROTATION", False),
    ("seis_prov:ATTENUATION", False)
))


trace = pr.entity("seis_prov:wf_A34J4DIDJ3", other_attributes=(
    ("prov:label", "Waveform Trace"),
    ("prov:type", "seis_prov:waveform_trace"),
))

simulation = pr.activity("seis_prov:ws_F87SF7SF78",
    startTime=datetime(2014, 2, 2, 12, 15, 3),
    endTime=datetime(2014, 2, 2, 14, 07, 13),
    other_attributes=(
    ("prov:label", "Waveform Simulation"),
    ("prov:type", "seis_prov:waveform_simulation"),
))


pr.association(simulation, specfem)
pr.association(model, other)
pr.delegation(specfem, me)

pr.usage(simulation, model)
pr.usage(simulation, param)

pr.generation(trace, simulation)
