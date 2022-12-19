UNLOAD (
    SELECT
            id,
            MAX(title) as title,
            MAX(abstract) as abstract,
            MAX(year) as year,
            ARRAY_JOIN(
                ARRAY_AGG(paragraph), chr(13) || chr(13)
            ) AS full_text,
            -- make 30 partitions for smaller output files
            floor(rand() * 30) as part_id
    FROM (
        SELECT
            id,
            year,
            title,
            abstract,
            fields_of_study,
            SUBSTR(full_text, paralocs[1], paralocs[2] - paralocs[1] + 1) as paragraph
        FROM (
            SELECT
                id,
                year,
                title,
                abstract,
                full_text,
                ARRAY_JOIN(fields_of_study, ',') as fields_of_study,
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
                ) AS all_paralocs
            FROM (
                SELECT
                    id,
                    metadata.publication_date.year as year,
                    metadata.title as title,
                    metadata.abstract as abstract,
                    CAST (
                        json_parse(metadata.fields_of_study) AS array(varchar)
                    ) AS fields_of_study,
                    content.grobid.contents as full_text,
                    CAST(
                        json_parse(content.grobid.annotations.paragraph)
                        AS array(json)
                    ) || CAST(
                        json_parse(content.grobid.annotations.section_header)
                        AS array(json)
                    ) AS all_paralocs
                FROM s2orc_papers.releases
                WHERE
                    year=2022 AND
                    month=12 AND
                    day=11 AND
                    metadata.fields_of_study IS NOT NULL
            )
            WHERE CONTAINS(fields_of_study, 'Computer Science')
        )
        CROSS JOIN UNNEST(all_paralocs) as t(paralocs)
    )
    GROUP BY id
)
TO 's3://ai2-s2-lucas/s2orc_20221211/cs_content/'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by = ARRAY['part_id']
)
