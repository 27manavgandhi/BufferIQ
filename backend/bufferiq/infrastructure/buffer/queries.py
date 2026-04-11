"""
GraphQL query and mutation definitions for Buffer API.

IMPORTANT: Buffer API uses custom scalar types:
- OrganizationId! (not ID!)
- ChannelId! (not ID!)
- PostId! (not ID!)
"""

# Organizations (via account)
GET_ORGANIZATIONS = """
query GetOrganizations {
  account {
    organizations {
      id
      name
    }
  }
}
"""

GET_ORGANIZATION = """
query GetOrganization($id: OrganizationId!) {
  organization(input: { id: $id }) {
    id
    name
  }
}
"""

# Channels
GET_CHANNELS = """
query GetChannels($organizationId: OrganizationId!) {
  channels(input: { organizationId: $organizationId }) {
    id
    name
    service
    avatar
    isQueuePaused
  }
}
"""

GET_CHANNEL = """
query GetChannel($id: ChannelId!) {
  channel(input: { id: $id }) {
    id
    name
    service
    displayName
    avatar
  }
}
"""

# Posts (cursor pagination)
GET_POSTS = """
query GetPosts($organizationId: OrganizationId!, $channelId: ChannelId!, $first: Int!) {
  posts(
    first: $first
    input: {
      organizationId: $organizationId
      filter: { channelIds: [$channelId] }
    }
  ) {
    edges {
      node {
        id
        text
        status
        dueAt
        channelId
        createdAt
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

GET_POST = """
query GetPost($id: PostId!) {
  post(input: { id: $id }) {
    id
    text
    status
    dueAt
    channelId
  }
}
"""

# Mutations
CREATE_POST = """
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    ... on PostActionSuccess {
      post {
        id
        text
        status
        dueAt
      }
    }
    ... on MutationError {
      message
    }
  }
}
"""

UPDATE_POST = """
mutation UpdatePost($input: UpdatePostInput!) {
  updatePost(input: $input) {
    ... on PostActionSuccess {
      post {
        id
        text
        status
        dueAt
      }
    }
    ... on MutationError {
      message
    }
  }
}
"""

DELETE_POST = """
mutation DeletePost($input: DeletePostInput!) {
  deletePost(input: $input) {
    ... on DeletePostSuccess {
      id
    }
    ... on MutationError {
      message
    }
  }
}
"""
