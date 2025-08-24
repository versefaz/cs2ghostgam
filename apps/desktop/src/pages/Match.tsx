import { gql, useMutation } from "@apollo/client";
import React from "react";

const ANALYZE = gql`
  mutation ($id:Int!){ analyze(matchId:$id){
    market selection stake_pct ev reasoning
  }}
`;

export default function MatchPage(){
  const [mutate, { data, loading, error }] = useMutation(ANALYZE);
  return (
    <div className="p-4">
      <button onClick={() => mutate({ variables: { id: 123 } })} className="px-3 py-2 bg-blue-600 text-white rounded">
        Analyze
      </button>
      {loading && <p>Loading...</p>}
      {error && <pre>{String(error)}</pre>}
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
    </div>
  );
}
