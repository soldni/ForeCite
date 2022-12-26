UNLOAD (
    SELECT
        id,
        year,
        title,
        abstract,
        fields_of_study,
        citations,
        ARRAY_JOIN(
            TRANSFORM(
                all_paralocs,
                paralocs -> NORMALIZE(
                    SUBSTR(
                        full_text,
                        paralocs[1],
                        paralocs[2] - paralocs[1] + 1
                    )
                )
            ),
        -- add two newlines between paragraphs
        chr(13) || chr(13)
        ) AS full_text,
        -- make 30 partitions for smaller output files
        floor(rand() * 30) AS part_id
    FROM (
        SELECT
            id,
            year,
            title,
            abstract,
            full_text,
            ARRAY_JOIN(fields_of_study, ',') AS fields_of_study,
            ARRAY_SORT(
                ARRAY_DISTINCT(
                    TRANSFORM(
                        all_paralocs,
                        x -> (
                            ARRAY [CAST(JSON_EXTRACT(x, '$.start') AS INT)] ||
                            ARRAY [CAST(JSON_EXTRACT(x, '$.end') AS INT)]
                        )
                    )
                )
            ) AS all_paralocs,
            FILTER(
                TRANSFORM(
                    bib_entries,
                    x -> CAST(
                        JSON_EXTRACT(x, '$.attributes.matched_paper_id')
                        AS BIGINT
                    )
                ),
                x -> IF(x IS NULL, False, True)
            ) AS citations
        FROM (
            SELECT
                id,
                metadata.publication_date.year AS year,
                metadata.title AS title,
                metadata.abstract AS abstract,
                CAST (
                    JSON_PARSE(metadata.fields_of_study) AS ARRAY(varchar)
                ) AS fields_of_study,
                content.grobid.contents AS full_text,
                CAST(
                    JSON_PARSE(content.grobid.annotations.paragraph)
                    AS ARRAY(json)
                ) || CAST(
                    JSON_PARSE(content.grobid.annotations.section_header)
                    AS ARRAY(json)
                ) AS all_paralocs,
                CAST(
                    JSON_PARSE(content.grobid.annotations.bib_entry)
                    AS ARRAY(json)
                ) as bib_entries
            FROM s2orc_papers.releases
            WHERE
                year=2022 AND
                month=12 AND
                day=11 AND
                metadata.fields_of_study IS NOT NULL
        )
        WHERE CONTAINS(fields_of_study, 'Computer Science')
    )
)
TO 's3://ai2-s2-lucas/s2orc_20221211/cs_content_cits/'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by = ARRAY['part_id']
)
